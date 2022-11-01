

https://download-directory.github.io/?url=https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection/dataset

mkdir clone && mkdir defect && mkdir translate && mkdir refine && mkdir generate && mkdir summarize
# apt-get install subversion

#clone
svn checkout https://github.com/microsoft/CodeXGLUE/trunk/Code-Code/Clone-detection-BigCloneBench/dataset
mv dataset clone
#defect
svn checkout https://github.com/microsoft/CodeXGLUE/trunk/Code-Code/Defect-detection/dataset
mv dataset defect && cd defect && python preprocess.py && cd ../
#summarize
svn checkout https://github.com/microsoft/CodeXGLUE/trunk/Code-Text/code-to-text/dataset.zip
unzip dataset.zip
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..


svn checkout https://github.com/microsoft/CodeXGLUE/trunk/Code-Code/Clone-detection-BigCloneBench/dataset
#translate
cd ../translate

#refine
cd ../refine
svn checkout https://github.com/microsoft/CodeXGLUE/trunk/Code-Code/code-refinement/data
mv data refine
#generate
cd ../generate
