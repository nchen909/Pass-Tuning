import logging
import torch
import argparse
import time
import multiprocessing
import os
import numpy as np
import math

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from configs import add_args, set_dist, set_seed, set_hyperparas
from models import bulid_or_load_gen_model,bulid_or_load_cls_model
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data, load_and_cache_defect_data,load_and_cache_clone_data, get_lang_by_task
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
import sys
from sklearn.metrics import recall_score, precision_score, f1_score
from tree_sitter import Language, Parser

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_cls(args, model, eval_examples, eval_data, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.dev_batch_size)

    # Eval!
    if write_to_pred == False:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Num batches = %d", len(eval_dataloader))
        logger.info("  Batch size = %d", args.dev_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        if args.sub_task == "POJ":
            inputs = batch[0].to(args.device)    
            p_inputs = batch[1].to(args.device)
            n_inputs = batch[2].to(args.device)
            label = batch[3].to(args.device)
        else:
            inputs = batch[0].to(args.device) # inputs shape: [batch_size, args.max_source_length+args.max_target_length]
            label = batch[1].to(args.device) # label shape: [batch_size]
        with torch.no_grad():
            if args.sub_task == "POJ":
                lm_loss, logit = model(inputs,p_inputs,n_inputs,label)
            else:
                lm_loss, logit = model(inputs, label) # logit shape:[batch_size]
            
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())#logit shape: [batch_size]
            labels.append(label.cpu().numpy())#label shape: [batch_size]
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    if args.model_name == "unixcoder" and args.task == "clone":
        preds = logits > 0.5
    else:
        preds = logits[:, 1] > 0.5
    if args.task == 'defect':
        eval_acc = np.mean(labels == preds)
        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.tensor(eval_loss)

        result = {
            "eval_loss": float(perplexity),
            "eval_acc": round(eval_acc, 4),
        }
    elif args.task == 'clone':
        recall = recall_score(labels, preds)
        precision = precision_score(labels, preds)
        f1 = f1_score(labels, preds)
        result = {
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1),
            "eval_threshold": 0.5,
        }
    if write_to_pred == False:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if write_to_pred:
        with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
            for example, pred in zip(eval_examples, preds):
                if args.task == 'defect':
                    if args.model_name == "unixcoder":
                        if pred:
                            f.write(str(example.idx) + '\t1\n')
                        else:
                            f.write(str(example.idx) + '\t0\n')
                    else:
                        if pred:
                            f.write(str(example.idx) + '\t1\n')
                        else:
                            f.write(str(example.idx) + '\t0\n')
                elif args.task == 'clone':
                    if pred:
                        f.write(example.url1 + '\t' + example.url2 + '\t' + '1' + '\n')
                    else:
                        f.write(example.url1 + '\t' + example.url2 + '\t' + '0' + '\n')

    return result

def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.dev_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.dev_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_name in ['roberta', 'codebert', 'graphcodebert']:
#                     loss, _, _, attention = model(source_ids=source_ids, source_mask=source_mask,
#                                           target_ids=target_ids, target_mask=target_mask)
                loss, _, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                    target_ids=target_ids, target_mask=target_mask)
                if args.n_gpu > 1:
                    loss = loss.mean()
                eval_loss += loss.item()
                batch_num += 1
            elif args.model_name in ['unixcoder']:
                _,loss,num = model(source_ids=source_ids,target_ids=target_ids)
                if args.n_gpu > 1:
                    loss = loss.mean()
                eval_loss += loss.sum().item()
                batch_num += num.sum().item()
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss
                if args.n_gpu > 1:
                    loss = loss.mean()
                eval_loss += loss.item()
                batch_num += 1
    
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info(
        "  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    if split_tag == 'dev':
        batch_size = args.dev_batch_size
    elif split_tag == 'test':
        batch_size = args.test_batch_size
    else:
        batch_size = args.batch_size
    logger.info("  Batch size = %d", batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device) #shape: (batch_size, max_source_len)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        if hasattr(model, 'module'):
            model = model.module # extract the model from the DataParallel wrapper
        with torch.no_grad():
            if args.model_name in ['roberta', 'codebert', 'graphcodebert']:
                preds, _ = model(source_ids=source_ids,
                                 source_mask=source_mask)
                top_preds = [pred[0].cpu().numpy() for pred in preds]
            elif args.model_name in ['unixcoder']:
                preds = model(source_ids=source_ids)  # preds shape: [batch_size, self.beam_size, max_target_len]
                top_preds = [pred[0].cpu().numpy() for pred in preds]# top_preds shape: batch_size * [max_target_len]
            else:
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)
    # pdb.set_trace()
    pred_nls = [tokenizer.decode(
        id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    # unixcoder in fewshot will generate '\n' in small batch, and gradually disappear
    if args.model_name in ['unixcoder']:
        pred_nls = [id.replace('\n','').replace("        "," ").replace("    "," ").replace("\t"," ") for id in pred_nls]
    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w',encoding='utf-8') as f, open(gold_fn, 'w',encoding='utf-8') as f1, open(src_fn, 'w',encoding='utf-8') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w',encoding='utf-8') as f, open(gold_fn, 'w',encoding='utf-8') as f1, open(src_fn, 'w',encoding='utf-8') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if args.task in ['summarize']:
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' +
                            pred_nl.strip().encode('utf8').decode() + '\n')
                    f1.write(str(gold.idx) + '\t' +
                             gold.target.strip().encode('utf8').decode() + '\n')
                    f2.write(str(gold.idx) + '\t' +
                             gold.source.strip().encode('utf8').decode() + '\n')
                else:
                    f.write(pred_nl.strip().encode('utf8').decode() + '\n')
                    f1.write(gold.target.strip().encode(
                        'utf8').decode() + '\n')
                    f2.write(gold.source.strip().encode(
                        'utf8').decode() + '\n')
        if args.task in ['summarize']:
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(
                goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if split_tag == 'test' and args.task in ['refine', 'translate', 'generate' , 'clone']:
                args.lang = get_lang_by_task(args.task, args.sub_task)
                codebleu = calc_code_bleu.get_codebleu(
                    gold_fn, output_fn, args.lang,args=args)
        # except:
        #     bleu = 0.0
        #     codebleu = 0.0

        em = np.mean(dev_accs) * 100
        result = {'em': em, 'bleu': bleu}
        if not args.task == 'summarize' and split_tag == 'test':
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    set_hyperparas(args)

    logger.info(args)
    logger.info("************* args: *************")
    logger.info("args.model_name: "+str(args.model_name))
    logger.info("args.few_shot: "+str(args.few_shot))
    logger.info("args.task: "+str(args.task))
    logger.info("args.sub_task: "+str(args.sub_task))
    logger.info("*********************************")
    
    if args.task in ['summarize', 'translate', 'refine', 'generate','complete']:
        config, model, tokenizer = bulid_or_load_gen_model(args)
    elif args.task in ['defect','clone']:
        config, model, tokenizer = bulid_or_load_cls_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_count)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(
        args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+',encoding='utf-8')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir,
                                        '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)
        # Prepare training data loader
        if args.task in ['summarize', 'translate', 'refine', 'generate','complete']:
            train_examples, train_data = load_and_cache_gen_data(
                args, args.train_filename, pool, tokenizer, 'train')
        elif args.task in ['defect']:
            train_examples, train_data = load_and_cache_defect_data(args, args.train_filename, pool, tokenizer, 'train')
        elif args.task in ['clone']:
            train_examples, train_data = load_and_cache_clone_data(args, args.train_filename, pool, tokenizer, 'train')

        train_sampler = RandomSampler(
            train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size,
                                      num_workers=4, pin_memory=True)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.lr, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * \
            len(train_dataloader)
        save_steps = max(len(train_dataloader), 1)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(args.warmup_steps) if args.warmup_steps >= 1 else num_train_optimization_steps * args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Batch num = %d", math.ceil(
            train_example_num / args.batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        if args.task in ['summarize', 'translate', 'refine', 'generate','complete']:
            dev_dataset = {}
            global_step, best_bleu_em, best_ppl = 0, -1, 1e6
            not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

            for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
                bar = tqdm(train_dataloader, total=len(
                    train_dataloader), desc="Training")
                nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
                model.train()
                for step, batch in enumerate(bar):
                    batch = tuple(t.to(args.device) for t in batch)

                    source_ids, target_ids = batch
                    source_mask = source_ids.ne(tokenizer.pad_token_id)
                    target_mask = target_ids.ne(tokenizer.pad_token_id)

                    if args.model_name in ['roberta', 'codebert', 'graphcodebert']:
    #                     loss, _, _, attention = model(source_ids=source_ids, source_mask=source_mask,
    #                                           target_ids=target_ids, target_mask=target_mask)
                        loss, _, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                            target_ids=target_ids, target_mask=target_mask)
                    elif args.model_name in ['unixcoder']:
                        loss,_,_ = model(source_ids=source_ids,target_ids=target_ids)#mask strategy (e2d) written in Seq2seq
                    else:
                        outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                        labels=target_ids, decoder_attention_mask=target_mask)
                        #attention =outputs.encoder_attentions
                        loss = outputs.loss
                    
                    
                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    tr_loss += loss.item()

                    nb_tr_examples += source_ids.size(0)
                    nb_tr_steps += 1
                    loss.backward()

                    if nb_tr_steps % args.gradient_accumulation_steps == 0:
                        # Update parameters
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        global_step += 1
                        train_loss = round(
                            tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                        bar.set_description("[{}] Train loss {}".format(
                            cur_epoch, round(train_loss, 3)))
        

                if args.do_eval:
                    # Eval model with dev dataset
                    if 'dev_loss' in dev_dataset:
                        eval_examples, eval_data = dev_dataset['dev_loss']
                    else:
                        eval_examples, eval_data = load_and_cache_gen_data(
                            args, args.dev_filename, pool, tokenizer, 'dev')
                        dev_dataset['dev_loss'] = eval_examples, eval_data

                    eval_ppl = eval_ppl_epoch(
                        args, eval_data, eval_examples, model, tokenizer)
                    result = {'epoch': cur_epoch,
                            'global_step': global_step, 'eval_ppl': eval_ppl}
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                    logger.info("  " + "*" * 20)
                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                    # save last checkpoint
                    if args.save_last_checkpoints:
                        last_output_dir = os.path.join(
                            args.output_dir, 'checkpoint-last')
                        if not os.path.exists(last_output_dir):
                            os.makedirs(last_output_dir)
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        output_model_file = os.path.join(
                            last_output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the last model into %s",
                                    output_model_file)

                    if eval_ppl < best_ppl:
                        not_loss_dec_cnt = 0
                        logger.info("  Best ppl:%s", eval_ppl)
                        logger.info("  " + "*" * 20)
                        fa.write("[%d] Best ppl changed into %.4f\n" %
                                (cur_epoch, eval_ppl))
                        best_ppl = eval_ppl

                        # Save best checkpoint for best ppl
                        output_dir = os.path.join(
                            args.output_dir, 'checkpoint-best-ppl')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.always_save_model:
                            model_to_save = model.module if hasattr(
                                model, 'module') else model
                            output_model_file = os.path.join(
                                output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(),
                                    output_model_file)
                            logger.info(
                                "Save the best ppl model into %s", output_model_file)
                    else:
                        not_loss_dec_cnt += 1
                        logger.info(
                            "Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(early_stop_str)
                            fa.write(early_stop_str)
                            break
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()
                    if args.do_eval_bleu:
                        eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                        only_src=True, is_sample=True)

                        result = eval_bleu_epoch(
                            args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                        dev_bleu, dev_em = result['bleu'], result['em']
                        if args.task in ['summarize']:
                            dev_bleu_em = dev_bleu
                        elif args.task in ['defect']:
                            dev_bleu_em = dev_em
                        else:
                            dev_bleu_em = dev_bleu + dev_em
                        if args.data_num == -1:
                            tb_writer.add_scalar(
                                'dev_bleu_em', dev_bleu_em, cur_epoch)
                            # tb_writer.add_scalar('dev_em', dev_em, cur_epoch)
                        if dev_bleu_em > best_bleu_em:
                            not_bleu_em_inc_cnt = 0
                            logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                        cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                            logger.info("  " + "*" * 20)
                            best_bleu_em = dev_bleu_em
                            fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                                cur_epoch, best_bleu_em, dev_bleu, dev_em))
                            # Save best checkpoint for best bleu
                            output_dir = os.path.join(
                                args.output_dir, 'checkpoint-best-bleu')
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            if args.data_num == -1 or args.always_save_model:
                                model_to_save = model.module if hasattr(
                                    model, 'module') else model
                                output_model_file = os.path.join(
                                    output_dir, "pytorch_model.bin")
                                torch.save(model_to_save.state_dict(),
                                        output_model_file)
                                logger.info(
                                    "Save the best bleu model into %s", output_model_file)
                        else:
                            not_bleu_em_inc_cnt += 1
                            logger.info(
                                "Bleu_em does not increase for %d epochs", not_bleu_em_inc_cnt)
                            if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                                stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                    cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                                logger.info(stop_early_str)
                                fa.write(stop_early_str)
                                break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
        
        elif args.task in ['defect']:
            global_step, best_acc = 0, 0
            not_acc_inc_cnt = 0
            is_early_stop = False
            for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
                bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
                nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
                model.train()
                for step, batch in enumerate(bar):
                    if args.sub_task == "POJ":
                        source_ids = batch[0].to(args.device)    
                        p_inputs = batch[1].to(args.device)
                        n_inputs = batch[2].to(args.device)
                        labels = batch[3].to(args.device)
                        model.train()
                        loss,vec = model(source_ids,p_inputs,n_inputs,labels)
                    else:
                        if args.model_name == 'unixcoder':
                            source_ids = batch[0].to(args.device)        
                            labels = batch[1].to(args.device) 
                            loss, logits = model(source_ids, labels)
                        else:
                            batch = tuple(t.to(args.device) for t in batch)
                            source_ids, labels = batch
                            loss, logits = model(source_ids, labels)

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    tr_loss += loss.item()

                    nb_tr_examples += source_ids.size(0)
                    nb_tr_steps += 1
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if nb_tr_steps % args.gradient_accumulation_steps == 0:
                        # Update parameters
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        global_step += 1
                        train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                        bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

                    if (step + 1) % save_steps == 0 and args.do_eval:
                        logger.info("***** CUDA.empty_cache() *****")
                        torch.cuda.empty_cache()

                        eval_examples, eval_data = load_and_cache_defect_data(args, args.dev_filename, pool, tokenizer,
                                                                            'dev',is_sample=True)

                        result = evaluate_cls(args, model, eval_examples, eval_data)
                        eval_acc = result['eval_acc']

                        if args.data_num == -1:
                            tb_writer.add_scalar('dev_acc', round(eval_acc, 4), cur_epoch)

                        # save last checkpoint
                        last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                        if not os.path.exists(last_output_dir):
                            os.makedirs(last_output_dir)

                        if True or args.data_num == -1 and args.save_last_checkpoints:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the last model into %s", output_model_file)

                        if eval_acc > best_acc:
                            not_acc_inc_cnt = 0
                            logger.info("  Best acc: %s", round(eval_acc, 4))
                            logger.info("  " + "*" * 20)
                            fa.write("[%d] Best acc changed into %.4f\n" % (cur_epoch, round(eval_acc, 4)))
                            best_acc = eval_acc
                            # Save best checkpoint for best ppl
                            output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            if args.data_num == -1 or True:
                                model_to_save = model.module if hasattr(model, 'module') else model
                                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                                torch.save(model_to_save.state_dict(), output_model_file)
                                logger.info("Save the best ppl model into %s", output_model_file)
                        else:
                            not_acc_inc_cnt += 1
                            logger.info("acc does not increase for %d epochs", not_acc_inc_cnt)
                            if not_acc_inc_cnt > args.patience:
                                logger.info("Early stop as acc do not increase for %d times", not_acc_inc_cnt)
                                fa.write("[%d] Early stop as not_acc_inc_cnt=%d\n" % (cur_epoch, not_acc_inc_cnt))
                                is_early_stop = True
                                break

                    model.train()
                if is_early_stop:
                    break

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
        elif args.task in ['clone']:
            global_step, best_f1 = 0, 0
            not_f1_inc_cnt = 0
            is_early_stop = False
            for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
                bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
                nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
                model.train()
                for step, batch in enumerate(bar):
                    if args.sub_task == "POJ":
                        source_ids = batch[0].to(args.device)
                        p_inputs = batch[1].to(args.device)
                        n_inputs = batch[2].to(args.device)
                        labels = batch[3].to(args.device)
                        loss,vec = model(source_ids,p_inputs,n_inputs,labels)
                    else:
                        if args.model_name == 'unixcoder':
                            source_ids = batch[0].to(args.device)        
                            labels = batch[1].to(args.device) 
                            loss, logits = model(source_ids, labels)
                        else:
                            batch = tuple(t.to(args.device) for t in batch)
                            source_ids, labels = batch #[batch,1024] [batch]
                            loss, logits = model(source_ids, labels)

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    tr_loss += loss.item()

                    nb_tr_examples += source_ids.size(0)
                    nb_tr_steps += 1
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if nb_tr_steps % args.gradient_accumulation_steps == 0:
                        # Update parameters
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        global_step += 1
                        train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                        bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

                    if (step + 1) % save_steps == 0 and args.do_eval:
                        logger.info("***** CUDA.empty_cache() *****")
                        torch.cuda.empty_cache()

                        eval_examples, eval_data = load_and_cache_clone_data(args, args.dev_filename, pool, tokenizer,
                                                                            'dev', is_sample=True)

                        result = evaluate_cls(args, model, eval_examples, eval_data)
                        eval_f1 = result['eval_f1']

                        if args.data_num == -1:
                            tb_writer.add_scalar('dev_f1', round(eval_f1, 4), cur_epoch)

                        # save last checkpoint
                        last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                        if not os.path.exists(last_output_dir):
                            os.makedirs(last_output_dir)

                        if True or args.data_num == -1 and args.save_last_checkpoints:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the last model into %s", output_model_file)

                        if eval_f1 > best_f1:
                            not_f1_inc_cnt = 0
                            logger.info("  Best f1: %s", round(eval_f1, 4))
                            logger.info("  " + "*" * 20)
                            fa.write("[%d] Best f1 changed into %.4f\n" % (cur_epoch, round(eval_f1, 4)))
                            best_f1 = eval_f1
                            # Save best checkpoint for best ppl
                            output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1')
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            if args.data_num == -1 or True:
                                model_to_save = model.module if hasattr(model, 'module') else model
                                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                                torch.save(model_to_save.state_dict(), output_model_file)
                                logger.info("Save the best ppl model into %s", output_model_file)
                        else:
                            not_f1_inc_cnt += 1
                            logger.info("F1 does not increase for %d epochs", not_f1_inc_cnt)
                            if not_f1_inc_cnt > args.patience:
                                logger.info("Early stop as f1 do not increase for %d times", not_f1_inc_cnt)
                                fa.write("[%d] Early stop as not_f1_inc_cnt=%d\n" % (cur_epoch, not_f1_inc_cnt))
                                is_early_stop = True
                                break

                    model.train()
                if is_early_stop:
                    break

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))
        


    if args.do_test:
        
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.test_batch_size)
        if args.task in ['summarize', 'translate', 'refine', 'generate','complete']:
            for criteria in ['best-bleu', 'best-ppl']:  # 'best-bleu', 'best-ppl', 'last'
                file = os.path.join(
                    args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
                logger.info("Reload model from {}".format(file))
                
                model = model.module if hasattr(model, 'module') else model
                model.load_state_dict(torch.load(file))
                # if args.n_gpu > 1:
                #     # multi-gpu training
                #     model = torch.nn.DataParallel(model)
                # model.load_state_dict(torch.load(file))
                eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                                only_src=True, is_sample=False)
                result = eval_bleu_epoch(
                    args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
                test_bleu, test_em = result['bleu'], result['em']
                test_codebleu = result['codebleu'] if 'codebleu' in result else 0
                result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (
                    criteria, test_bleu, test_em, test_codebleu)
                logger.info("*********************************")
                logger.info(result_str)
                logger.info("*********************************")
                fa.write(result_str)
                if args.res_fn:
                    with open(args.res_fn, 'a+',encoding='utf-8') as f:
                        f.write('[Time: {}] {}\n'.format(
                            get_elapse_time(t0), file))
                        f.write(result_str)
        elif args.task in ['defect']:
            for criteria in ['best-acc']:
                file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
                logger.info("Reload model from {}".format(file))

                model = model.module if hasattr(model, 'module') else model
                model.load_state_dict(torch.load(file))
                # if args.n_gpu > 1:
                #     # multi-gpu training
                #     model = torch.nn.DataParallel(model)

                eval_examples, eval_data = load_and_cache_defect_data(args, args.test_filename, pool, tokenizer, 'test',
                                                                    False)
                result = evaluate_cls(args, model, eval_examples, eval_data, write_to_pred=True)
                logger.info(" test_acc = %.4f" % ( result['eval_acc']))
                logger.info("*********************************")
                logger.info("[%s]  test_acc = %.4f" % (criteria, result['eval_acc']))
                logger.info("*********************************")

                fa.write("[%s] test-acc: %.4f\n" % (criteria, result['eval_acc']))
                if args.res_fn:
                    with open(args.res_fn, 'a+') as f:
                        f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                        f.write("[%s] acc: %.4f\n\n" % (
                            criteria, result['eval_acc']))
        elif args.task in ['clone']:
            for criteria in ['best-f1']:
                file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
                logger.info("Reload model from {}".format(file))
                
                model = model.module if hasattr(model, 'module') else model
                model.load_state_dict(torch.load(file))
                # if args.n_gpu > 1:
                #     # multi-gpu training
                #     model = torch.nn.DataParallel(model)
                # model.load_state_dict(torch.load(file))

                eval_examples, eval_data = load_and_cache_clone_data(args, args.test_filename, pool, tokenizer, 'test',
                                                                    False)

                result = evaluate_cls(args, model, eval_examples, eval_data, write_to_pred=True)
                
                logger.info("  test_f1 = %.4f", result['eval_f1'])
                logger.info("  test_prec = %.4f", result['eval_precision'])
                logger.info("  test_rec = %.4f", result['eval_recall'])
                logger.info("*********************************")
                logger.info("[%s] test-f1: %.4f, precision: %.4f, recall: %.4f\n" % (
                    criteria, result['eval_f1'], result['eval_precision'], result['eval_recall']))
                logger.info("*********************************")

                fa.write("[%s] test-f1: %.4f, precision: %.4f, recall: %.4f\n" % (
                    criteria, result['eval_f1'], result['eval_precision'], result['eval_recall']))
                if args.res_fn:
                    with open(args.res_fn, 'a+') as f:
                        f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                        f.write("[%s] f1: %.4f, precision: %.4f, recall: %.4f\n\n" % (
                            criteria, result['eval_f1'], result['eval_precision'], result['eval_recall']))

    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
