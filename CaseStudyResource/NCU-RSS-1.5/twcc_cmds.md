# TWCC ssh commands
## twcc login passw
```
??
??
```
[ip]=??
[port]=??

# ssh connection
注意[port]號碼、[ip]每次都會變
```sh
ssh u2443177@[ip] -p [port]
```
# upload
## upload training images
```sh
scp -r -P [port] D:\1112_work\'230412_國網中心training產生之模型效能不佳之除錯'\flow_b_output_from_twcc\*.zip u2443177@[ip]:/work/u2443177/Perry/test_rss13_230417

```

## upload py files
```sh
scp -r -P [port] D:\1111_work\NCU-RSS-1.3\*.py u2443177@[ip]:/work/u2443177/Perry/test_rss13_230417/

```

## upload src
```sh
scp -r -P [port] D:\1111_work\NCU-RSS-1.3\src u2443177@[ip]:/work/u2443177/Perry/test_rss13_230417/

```

## upload config
```sh
scp -r -P [port] D:\1111_work\NCU-RSS-1.3\configs u2443177@[ip]:/work/u2443177/Perry/test_rss13_230417/

```
## upload inference images
```sh
scp -r -P [port] D:\1112_work\230412_國網中心training產生之模型效能不佳之除錯\inference_img_framelets_94201090\IMG_CROP.zip u2443177@[ip]:/work/u2443177/Perry/test_rss13_230417

```

# train
```
python train.py
```
# download results
## prediction_img
```sh
scp -r -P [port] u2443177@[ip]:/work/u2443177/Perry/test_rss13_230417/data/*.png D:\1112_work\230412_國網中心training產生之模型效能不佳之除錯\incoming
```
## fold_1_results
```sh
scp -r -P [port] u2443177@[ip]:/work/u2443177/Perry/test_rss13_230417/data/fold_1_results D:\1112_work\230412_國網中心training產生之模型效能不佳之除錯\incoming
```

# other useful commands
## get file count
```sh
ls -l Perry/ | wc -l
```

## trash 
```sh
/home/u2443177/.local/share/Trash/files/
```

## check disk usage
```sh
du -sh ./files
```
## tar
```sh
tar -zcf#compress
tar -zxf#extract
```