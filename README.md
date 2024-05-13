# Chinese-Sentiment-Analysis-Fund-Direction


    验证集准确度: 0.9382193411826961
    验证集分类报告: 
                  precision    recall  f1-score   support
    
        negative       0.93      0.95      0.94      3785
        positive       0.95      0.96      0.95      6919
         neutral       0.93      0.89      0.91      4414
    
        accuracy                           0.94     15118
       macro avg       0.94      0.93      0.93     15118
    weighted avg       0.94      0.94      0.94     15118

  大概使用了10w+的数据做了一个基金方面的中文情感分析模型，暂时测试下来还可以，负面方面的文本是有专人处理过的，中性的可能不准确。
# 返回值解释：
  0: 'negative', 1: 'positive', 2: 'neutral'



# 测试代码如下：
    import sys
    import re
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.nn.functional import softmax
    
    #设定使用CPU或CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #载入预先保存的模型和分词器
    model = BertForSequenceClassification.from_pretrained('sanshizhang/Chinese-Sentiment-Analysis-Fund-Direction')
    tokenizer = BertTokenizer.from_pretrained('sanshizhang/Chinese-Sentiment-Analysis-Fund-Direction')
    
    #确保模型在正确的设备上
    model = model.to(device)
    model.eval()  # 把模型设置为评估模式
    
    #函数定义：进行预测并返回预测概率
    def predict_sentiment(text):
        # 编码文本数据
        encoding = tokenizer.encode_plus(
            text,
            max_length=512,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',  # 修改此处
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        
        # 取出输入对应的编码
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # 不计算梯度
        with torch.no_grad():
        # 产生情感预测的logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用softmax将logits转换为概率
        probs = softmax(outputs.logits, dim=1)
    
        # 返回概率和预测的类别
        return probs, torch.argmax(probs, dim=1).cpu().numpy()[0]
    
    #从命令行参数获取文本，合并并清理特殊字符
    arguments = sys.argv[1:]  # 忽略脚本名称
    text = ' '.join(arguments)  # 合并为单一字符串
    text = re.sub(r"[^\u4e00-\u9fff\d.a-zA-Z%+\-。！？，、；：（）【】《》“”‘’]", '', text)  # 去除特殊字符
    
    #print(f"传过来的文本是: {text}")
    #进行预测
    probabilities, prediction = predict_sentiment(text)
    
    sentiment_labels = {0: 'negative', 1: 'positive', 2: 'neutral'}
    
    #打印出预测的情感及其概率
    predicted_sentiment = sentiment_labels[prediction]
    print(f"Predicted sentiment: {predicted_sentiment},Probability:{probabilities[0][prediction].item()}")
    #print(f"Probability: {probabilities[0][prediction].item()}")
