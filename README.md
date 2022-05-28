# Colorectal cancer cells segmentation on histological images

Colorectal cancer is one of the three most common types of cancer in the world. Histological and immunohistochemical examination of the tumor tissue taken with the help of a biopsy is the standard for making a diagnosis and choosing the correct treatment plan.

Based on a dataset of histological images of malignant neoplasms of the colon, rectosigmoid and rectum, a semantic segmentation model was developed that is able to separate and label healthy/cancerous tissues.

<img src="docs/18-04842B_2019-05-07 23_40_49-lv1-22633-15157-4325-3516.jpg" alt="" align="middle" width="600"/>

## Run pipeline & tensorboard
~~~
python src/main.py config.txt
tensorboard --logdir runs
~~~

<img src="docs/runs.jfif" alt="" align="middle" width="600"/>
