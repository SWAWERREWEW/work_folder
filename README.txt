Удаление файлов и папки из отслеживаемых
git rm --cached .idea/vcs.xml

git rm --cached .idea/my_learning_django_2500_stepik_id_2135123512525235110916.iml

git rm --cached .idea/misc.xml

git rm --cached sitewomen\db.json

git rm -r --cached .idea

откатить коммит, сохранив изменения в индексе
git reset --soft HEAD~1

убрать файл из индекса
git reset HEAD sitewomen\sitewomen\settings.py
git reset HEAD sitewomen\db.json

git reset HEAD 

создать новый коммит с тем же сообщением
git commit -c ORIG_HEAD

выход из окна редактирования
:wq

отправка на github
git push origin main --force
