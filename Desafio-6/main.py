#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)


# In[ ]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[ ]:


countries = pd.read_csv("countries.csv")


# In[ ]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[ ]:


# Sua análise começa aqui.
countries.dtypes


# In[ ]:


countries.columns


# In[ ]:


for col in countries.select_dtypes(include = 'object').columns:
        countries[col] = countries[col].str.replace(',', '.')


# In[ ]:


for col in new_column_names[2:]:
        countries[col] = countries[col].astype('float64')


# In[ ]:


countries.dtypes


# In[ ]:


countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()


# In[ ]:


countries.head()


# In[ ]:


countries.isnull().sum()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[ ]:


def q1():
    # Retorne aqui o resultado da questão 1.
    reg_unique = countries['Region'].sort_values().unique()
    return list(reg_unique)
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[ ]:


def q2():
    # Retorne aqui o resultado da questão 2.
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    score_bins = discretizer.fit_transform(countries[['Pop_density']])
    return sum(score_bins[:, 0] == 9)
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[ ]:


def q3():
    # Retorne aqui o resultado da questão 3. 
    encoded_features = OneHotEncoder(sparse=False).fit_transform(countries[['Region', 'Climate']].fillna(0))
    return encoded_features.shape[1]
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[ ]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[ ]:


def q4():
    # Retorne aqui o resultado da questão 4.
    df = pd.DataFrame([test_country], columns=countries.columns)
    columns = countries.columns[2:]
    pipeline_cols = Pipeline(steps=[('inputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    pipe = pipeline_cols.fit(countries[columns])
    transform_test_country = pipe.transform(df[columns])
    df_country_pipeline_transform = pd.DataFrame(transform_test_country, columns=columns)
    return float(round(df_country_pipeline_transform.Arable,3))
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[ ]:


def q5():
    # Retorne aqui o resultado da questão 5.
    quantile_1 = countries['Net_migration'].quantile(0.25)
    quantile_3 = countries['Net_migration'].quantile(0.75)
    iqr = quantile_3 - quantile_1
    interval_iqr = [quantile_1 -1.5 * iqr, quantile_3 + 1.5 * iqr]
    low_outlier = countries[(countries['Net_migration'] < interval_iqr[0])]
    high_outlier = countries[(countries['Net_migration'] > interval_iqr[1])]
    remove = False
    return tuple([len(low_outlier), len(high_outlier), remove])
q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[ ]:


def q6():
    # Retorne aqui o resultado da questão 6.
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    count_vectorizer = CountVectorizer()
    new_transform = count_vectorizer.fit_transform(newsgroup.data)
    word = 'phone'
    phone_idx = count_vectorizer.vocabulary_.get(word.lower())
    return int(new_transform[:, phone_idx].toarray().sum())
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[ ]:


def q7():
    # Retorne aqui o resultado da questão 7.
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    count_vectorizer = CountVectorizer()
    newsgroup_count = count_vectorizer.fit_transform(newsgroup.data)
    word = 'phone'
    phone_idx = count_vectorizer.vocabulary_.get(word.lower())
    tfidf_vectorized = TfidfVectorizer()
    tfidf_vectorized.fit(newsgroup.data)
    newsgroup_tfidf_vectorized = tfidf_vectorized.transform(newsgroup.data)
    return(float(np.round(newsgroup_tfidf_vectorized[:, phone_idx].toarray().sum(),3)))
q7()


# In[ ]:




