# Rechnerarchitekturen für Deep-Learning Anwendungen
You can find the Source Code README [here](./Source%20Code/README.md)
```sh
NAME="Robert Kagan"
EMAIL=robert.kagan@fau.de
USER=cu14mowo

git config --global user.name "$NAME"
git config --global user.email $EMAIL
git config --global core.editor nano
git config --global init.defaultBranch main

mkdir -p ~/.ssh
ssh-keygen -C $EMAIL -f ~/.ssh/fau -N ""
echo -e "\n>>> https://gitlab.cs.fau.de/-/user_settings/ssh_keys <<<"
cat ~/.ssh/fau.pub

echo "Host gitlab.cs.fau.de" > ~/.ssh/config
echo -e "\tUser $USER" >> ~/.ssh/config
echo -e "\tPreferredAuthentications publickey" >> ~/.ssh/config
echo -e "\tIdentityFile ~/.ssh/fau" >> ~/.ssh/config
```
```sh
git clone git@gitlab.cs.fau.de:cu14mowo/radl.git "Rechnerarchitekturen für Deep-Learning Anwendungen"
```
```sh
git rebase -i HEAD~X
# X is the number of commits to go back
# Move to the line of your commit, change pick into reword,
# then change your commit message:
git commit --amend --author="$NAME <$EMAIL>" --no-edit
# Finish the rebase with:
git rebase --continue
git push --force
```
```sh
git pull --rebase origin main
git push --tags
```
