# Compound Vowel Splitting Guide / 复合元音拆分使用指南

[English](#english) | [简体中文](#简体中文)

## English

### Introduction

This guide explains how to use the compound vowel splitting feature in SOFA. This feature allows you to train a model that can split compound vowels (diphthongs) into their component simple vowels and use it for forced alignment.

### Workflow Overview

1. **Prepare split rules dictionary**: Define how compound vowels should be split
2. **Prepare training data**: Create training data with split phoneme annotations
3. **Train the model**: Train a model to learn split points
4. **Run split inference**: Use the trained model to split compound vowels in new audio

### Step 1: Create Split Rules Dictionary

Create a dictionary file that defines how compound vowels should be split. The format is:
```
compound_vowel<TAB>component1 component2 [component3 ...]
```

Example (`dictionary/vowel_split_example.txt`):
```
ai	a i
ao	a o
ei	e i
ou	o u
iao	i a o
```

### Step 2: Prepare Training Data

You have two options:

#### Option A: Use existing annotations with automatic conversion

If you have existing training data with compound vowels, use the conversion script:

```bash
python prepare_split_data.py \
    --input data/full_label \
    --output data/split_label \
    --split_dictionary dictionary/vowel_split_example.txt \
    --copy_wavs
```

This will:
- Read existing transcriptions.csv files
- Split compound vowels according to your rules
- Distribute the original duration equally among component vowels
- Write converted files to the output directory

#### Option B: Manually annotate split data

Manually create transcriptions.csv with split phoneme sequences. For example:
```csv
name,ph_seq,ph_dur
sample1,a i,0.1 0.1
sample2,SP a o SP,0.05 0.1 0.1 0.05
```

### Step 3: Binarize and Train

1. Binarize the training data:
```bash
python binarize.py -c configs/split_binarize_config.yaml
```

2. Train the model:
```bash
python train.py -c configs/split_train_config.yaml -p pretrained_model.ckpt
```

### Step 4: Run Split Inference

Use the trained model with split inference:

```bash
# Using the dedicated split inference script
python split_infer.py \
    --ckpt ckpt/split_model.ckpt \
    --folder segments \
    --split_dictionary dictionary/vowel_split_example.txt \
    --dictionary dictionary/opencpop-extension.txt

# Or using the main inference script with Split G2P
python infer.py \
    --ckpt ckpt/split_model.ckpt \
    --folder segments \
    --g2p Split \
    --split_dictionary dictionary/vowel_split_example.txt \
    --dictionary dictionary/opencpop-extension.txt
```

### Output

The output will contain phoneme-level alignments where compound vowels are split into their component vowels. For example, if the input contains "ai", the output will show separate time intervals for "a" and "i".

---

## 简体中文

### 介绍

本指南介绍如何使用SOFA的复合元音拆分功能。该功能允许你训练一个模型，将复合元音（双元音）拆分成简单元音，并用于强制对齐。

### 工作流程概述

1. **准备拆分规则字典**：定义复合元音如何拆分
2. **准备训练数据**：创建带有拆分音素标注的训练数据
3. **训练模型**：训练模型学习拆分点
4. **运行拆分推理**：使用训练好的模型在新音频中拆分复合元音

### 步骤1：创建拆分规则字典

创建一个定义复合元音拆分方式的字典文件。格式为：
```
复合元音<制表符>成分1 成分2 [成分3 ...]
```

示例（`dictionary/vowel_split_example.txt`）：
```
ai	a i
ao	a o
ei	e i
ou	o u
iao	i a o
```

### 步骤2：准备训练数据

有两种选择：

#### 选项A：使用现有标注自动转换

如果你有现有的带复合元音的训练数据，使用转换脚本：

```bash
python prepare_split_data.py \
    --input data/full_label \
    --output data/split_label \
    --split_dictionary dictionary/vowel_split_example.txt \
    --copy_wavs
```

这将：
- 读取现有的transcriptions.csv文件
- 根据规则拆分复合元音
- 将原始时长平均分配给各成分元音
- 将转换后的文件写入输出目录

#### 选项B：手动标注拆分数据

手动创建带有拆分音素序列的transcriptions.csv。例如：
```csv
name,ph_seq,ph_dur
sample1,a i,0.1 0.1
sample2,SP a o SP,0.05 0.1 0.1 0.05
```

### 步骤3：二值化和训练

1. 二值化训练数据：
```bash
python binarize.py -c configs/split_binarize_config.yaml
```

2. 训练模型：
```bash
python train.py -c configs/split_train_config.yaml -p pretrained_model.ckpt
```

### 步骤4：运行拆分推理

使用训练好的模型进行拆分推理：

```bash
# 使用专用的拆分推理脚本
python split_infer.py \
    --ckpt ckpt/split_model.ckpt \
    --folder segments \
    --split_dictionary dictionary/vowel_split_example.txt \
    --dictionary dictionary/opencpop-extension.txt

# 或使用主推理脚本配合Split G2P
python infer.py \
    --ckpt ckpt/split_model.ckpt \
    --folder segments \
    --g2p Split \
    --split_dictionary dictionary/vowel_split_example.txt \
    --dictionary dictionary/opencpop-extension.txt
```

### 输出

输出将包含音素级别的对齐，其中复合元音被拆分成其成分元音。例如，如果输入包含"ai"，输出将显示"a"和"i"的单独时间间隔。

### 注意事项

- 确保拆分后的简单元音存在于你的模型词汇表中
- 如果你的模型是在复合元音上训练的，需要重新训练以支持拆分
- 拆分规则应该与你的语音学分析一致
