#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import cv2
import pytesseract
from glob import glob
import spacy
import re
import string


# In[26]:


def cleanText(txt):
    whitespace = string.whitespace
    punctuation = '!#$%&\'()*+-:;<=>?[\\]^`{|}~'
    tableWhitespace = str.maketrans('','', whitespace)
    tablePunctuation = str.maketrans('', '', punctuation)
    text = str(txt)
    text = text.lower()
    removewhitespace = text.translate(tableWhitespace)
    removepunctuation = removewhitespace.translate(tablePunctuation)

    return str(removepunctuation)


# In[27]:


### Load NER Model
model_ner = spacy.load('./output/model-best/')


# In[28]:


# Load Image
image = cv2.imread('./data/041.jpeg')

# cv2.imshow('businessCard', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows

# extract data using PyTesseract
tessData = pytesseract.image_to_data(image)
tessData

# convert into dataframe
tessList = list(map(lambda x: x.split('\t'),tessData.split('\n')))
df = pd.DataFrame(tessList[1:], columns=tessList[0])
df.dropna(inplace=True)
df['text'] = df['text'].apply(cleanText)

# convert data into content
df_clean = df.query('text != "" ')
content = " ".join([w for w in df_clean['text']])
# print(content)


# get predictions from NER model
doc = model_ner(content)


# In[33]:


from spacy import displacy


# In[34]:


# displacy.serve(doc, style='ent')


# In[35]:


displacy.render(doc, style='ent')


# In[36]:


### Tagging


# In[38]:


docjson = doc.to_json()
docjson.keys()


# In[41]:


doc_text = docjson['text']
doc_text


# In[46]:


datafram_tokens = pd.DataFrame(docjson['tokens'])
datafram_tokens['token'] = datafram_tokens[['start', 'end']].apply(
    lambda x: doc_text[x[0]:x[1]], axis = 1
)

datafram_tokens.head(10)


# In[51]:


right_table = pd.DataFrame(docjson['ents'])[['start', 'label']]
datafram_tokens = pd.merge(datafram_tokens, right_table, how='left', on='start')


# In[53]:


datafram_tokens.fillna('O', inplace=True)
datafram_tokens.head(10)


# In[60]:


# join label to df_clean dataframe
df_clean['end'] = df_clean['text'].apply(lambda x: len(x)+1).cumsum() - 1
df_clean['start'] = df_clean[['text', 'end']].apply(lambda x: x[1] - len(x[0]), axis = 1)


# In[62]:


# inner join with start
dataframe_info = pd.merge(df_clean, datafram_tokens[['start', 'token', 'label']], how='inner', on='start')


# In[67]:


dataframe_info.tail(10)

### Bounding Box
# In[70]:


bb_df = dataframe_info.query("label !='O'")
img = image.copy()

for x,y,w,h,label in bb_df[['left', 'top', 'width', 'height', 'label']].values:
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    cv2.rectangle(img,(x,y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, str(label), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

cv2.imshow('Predictions', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[71]:


bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])
bb_df.head()


# In[80]:


#group the label
class groupgen():
    def __init__(self):
        self.id = 0
        self.text = ''

    def getgroup(self, text):
        if self.text == text:
            return self.id
        else:
            self.id += 1
            self.text = text
            return self.id
grp_gen = groupgen()


# In[81]:


bb_df['group'] = bb_df['label'].apply(grp_gen.getgroup)


# In[82]:


# right and bottom of bounding box
bb_df[['left', 'top', 'width', 'height']] = bb_df[['left', 'top', 'width', 'height']].astype(int)
bb_df['right'] = bb_df['left'] + bb_df['width']
bb_df['bottom'] = bb_df['top'] + bb_df['height']


# In[83]:


bb_df


# In[84]:


# tagging: groupby group
col_group = ['left', 'top', 'right', 'bottom', 'label', 'token', 'group']
group_tag_img = bb_df[col_group].groupby(by='group')


# In[85]:


img_tagging = group_tag_img.agg({
    'left':min,
    'right':max,
    'top':min,
    'bottom':max,
    'label':np.unique,
    'token':lambda x: " ".join(x)
})


# In[86]:


img_tagging


# In[91]:


img_bb = image.copy()
for l,r,t,b,label,token in img_tagging.values:
    label = str(label)
    cv2.rectangle(img_bb, (l, t), (r,b), (0, 255, 0), 2)
    cv2.putText(img_bb, label, (l,t), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

cv2.imshow("Bounding Box BusinessCard", img_bb)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[92]:


### PARSER


# In[123]:


def parser(text, label):
    if label == 'PHONE':
        text = text.lower()
        text = re.sub(r'\D','',text)
        
    elif label == 'EMAIL':
        text = text.lower()
        allow_special_chars = '@_.\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_chars), '', text)

    elif label == 'WEB':
        text = text.lower()
        allow_special_chars = ':/.%#\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_chars), '', text)

    elif label in ('NAME', 'DES'):
        text = text.lower()
        text = re.sub(r'[^a-z ]', '', text)
        text = text.title()

    elif label == 'ORG':
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = text.title()
    return text


# In[124]:


parser('Srikanth)&#$@gmail.COM', 'WEB')


# In[119]:


### ENTITIES


# In[125]:


info_array = dataframe_info[['token', 'label']].values
entities = dict(NAME=[],ORG=[],DES=[],PHONE=[],EMAIL=[],WEB=[])
previous = 'O'

for token, label in info_array:
    bio_tag = label[:1]
    label_tag = label[2:]

    # step-1 - parse the token
    text = parser(token, label_tag)

    if bio_tag in ('B', 'I'):
        if previous != label_tag:
            entities[label_tag].append(text)
        
        else:
            if bio_tag == "B":
                entities[label_tag].append(text)
    
            else:
                if label_tag in ("NAME", "ORG", "DES"):
                    entities[label_tag][-1] = entities[label_tag][-1] + " " + text
    
                else:
                    entities[label_tag][-1] = entities[label_tag][-1] + text

    previous = label_tag


# In[127]:


entities


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




