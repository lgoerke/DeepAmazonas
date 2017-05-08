# DeepAmazonas


## How to download data to surfsara

* Get browser add-on to download cookies as txt
* Log into kaggle
* In browser go to tool/add-ons and export cookies
* open terminal and navigate to cookies.txt
* scp gdemo'you-user'@cartesius.surfsara.nl:.
* ssh gdemo'you-user'@cartesius.surfsara.nl
* On kaggle go to Data
* Click on dataset you want to download
* Righ-click onto 'Download File' and select 'Copy Link Location'
* On Cartesius execute 'wget --load-cookies cookies.txt 'copied-link-location''
* To unzip 7z execute 'module load p7zip; 7za x 'zipped directory''
