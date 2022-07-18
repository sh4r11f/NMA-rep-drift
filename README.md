# NMA-rep-drift
Repo for analysis of representational drift with Allen institute data (NMA Sachertorte Pod Group 1)

## Mini Github colaboration guide

First, to make a local copy of the repo, past the below code in command line:

```
$git clone https://github.com/DevXl/NMA-rep-drift.git
```

Branches are usually representing features, e.g., when we want to add a new function block like "low dim trajectory analysis", we should create a branch with such name and in that branch, only update that part.

- Discuss: maybe we should first have one or two of us working on setting up the main, and we all can pull that version.

When we want to create a new branch, in terminal:

```
$git checkout -b tj_analysis
```

`checkout my_branch` alone allows switching to  `my_branch` to work on. Adding the `-b` and branch name at the end creates a new branch and then moves into that new branch.

Can verify checking out new branch with the command:

```
$ git branch
```

Now you can start coding in your own branch!

When you finish up a function, or something that you want to save as a version to be able to go back to, do

```
git add .
```

which add all changes you made before your last commit to the staging area.

and then do:

```
git commit -m 'added xxx analysis features`
```

to save the changes in the staging area; and you can use `git status` to check what modification has been staged.

-----

The first time you want to push the local changes to your own branch in the reomote repository, do

```
git push --set-upstream origin remote-branch
```

You can check on the github page if the changes are reflected.

- Discuss: shall we choose the `pull request` route and review each other's code before we merge?

-----


## Mini function sharing guide

When we code on a notebook, we can always convert it to a python script by download it as a python file:

1. In Jupyter lab, go to File Menu. Select "Export Notebook as" and then choose the "Export Notebook to Executable Script" option.

2. For Jupyter notebook, there is "Download as" option in File menu of Jupyter notebook. You can select Python (.py) from the list of available options.

## Access data folders in Google Drive from a Colab notebook:


``` python 
from google.colab import drive

# allow Colab to access google drive:
drive.mount('/content/drive')

file_path = 'drive/My Drive/data.csv'
```
