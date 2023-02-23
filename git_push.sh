git init
gh repo create ML_Example --public
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YiyaoZhang9527/ML_Example.git
git push -u origin master


# git rm -r --cached .
# rm .git/index
# git commit -m "update commit 1"
# git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch NLP数据集合/词性标注数据集/2014/2014_corpus.txt'
# git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch NLP/model.ckpt'
# git push -u origin master -f