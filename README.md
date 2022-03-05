Merge lines and build graphs after HoughLinesP.
<a name="a3DFB"></a>
## Problem Description
I have been working for adding the road extract function to my Data Structure homework for last week. There are many neural networks about inferrring road graphs from satellite imagery on the Internet so it is easy to find. However, I found few projects on merging lines and building graphs through the extracted gary images, and HoughLinesP is not perfect in this project because there are still many short and similar lines even if you have tested a large number of parameters . In this case, I designed this program for further processing.
<a name="k4JMk"></a>
## Road Extraction Network
[D-LinkNet](https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge), and Their [Paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.html)
<a name="sJ9qn"></a>
## File Structure
networks

- dinknet.py

submit

- img.jpg(the image you want to deal with, one image only)

submits

- data.txt
- mask.png
- merged_lines.jpg

(all above are the output of this program )<br />weighs

- first.th

(you can download this and images from [here](https://pan.baidu.com/s/1SZLPor4Z008unO9cy6fUXg?pwd=8g56))<br />Merge.py<br />README.md
<a name="uuquO"></a>
## Show
![image.png](https://cdn.nlark.com/yuque/0/2022/png/2677447/1646479613309-2efd9956-1892-41ea-aced-025fcde0a8a8.png#clientId=ubdbe1eac-43f2-4&from=paste&height=780&id=ka7D8&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1560&originWidth=2394&originalType=binary&ratio=1&size=3364870&status=done&style=none&taskId=u0d06a941-a270-4123-bde8-76aca461404&width=1197)
<a name="wM4pv"></a>
## Usage

1. Download the weighs and images ([here](https://pan.baidu.com/s/1SZLPor4Z008unO9cy6fUXg?pwd=8g56))
1. Place the "first.th" in the "weights" folder
1. Place only one image in the "submit" folder
1. Run Merge.py , get the results in the "submits" folder


<br />
<br />
<br />
<br />​<br />
