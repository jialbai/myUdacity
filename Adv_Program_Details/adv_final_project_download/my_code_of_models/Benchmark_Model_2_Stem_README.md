# Benchmark_Model_2_stem.py之前的准备工作：


## 需要准备的文件

1.	下载 20news-bydate 数据集 (训练集+测试集)
2.	导入环境文件 JialiangBAI_TextCNN_LR.yaml
3.	安装文件预处理工具nltk，并下载stopwords和punkt两个组件
	import nltk
	nltk.download('stopwords')
	nltk.download('punkt')

## 运行该代码时的目录结构

.
+-- Benchmark_Model_2_stem.py
+-- 20news-bydate
|   +-- 20news-bydate-test
|   +--   +-- alt.atheism
|   +--   +-- comp.graphics
|   +--   +-- comp.os.ms-windows.misc
|   +--   +-- ... ... 
|   +-- 20news-bydate-train
|   +--   +-- alt.atheism
|   +--   +-- comp.graphics
|   +--   +-- comp.os.ms-windows.misc
|   +--   +-- ... ... 
