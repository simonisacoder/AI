<img src="img/logo.png">
***
<h2 align=center>��ɽ��ѧ���ݿ�ѧ������ѧԺ</h2>
<h2 align=center>�ƶ���Ϣ���� - �˹�����</h2>
<h2 align=center>������ʵ�鱨��</h2>
<h6 align=center>(2017-2018ѧ���＾ѧ��)</h6>


|��ѧ�༶|רҵ������|ѧ��|����|
|------ | ------ |--|--|
|1513(M3)|�ƶ���Ϣ���̣���������|15352296|�ռ���


## һ��ʵ����Ŀ
�ı����ݼ��ļ򵥴���

## ����ʵ������
### �ٳ�ȡ��������
˼·��ǰ���������Ǽ��һ�����������Ƿ����֣�ǰһ��ʹ��any�ķ�������һ��ʹ��������ʽ����������ȥû�õĴ���
</br>
���һ��������ͨ������һ����ά�б���������Ϣ���������棬�Լ�һ��һά�б����ճ���˳��������е���
```
# any������鵥���Ƿ�����
def hasNum1(string):
    return any(char.isdigit() for char in string)

#������ʽ����Ƿ�����
def hasNum2(string):
    return bool(re.search(r'\d',string))
```
```
def substract_data():
    f = open('semeval','r')
    line = f.readline()
    #������ݵ�һ���б�
    data_list = []

    #key code-------------------------------
    # ���ж�ȡ�ļ�
    while line:
        #�Ѵ�����ֳ���
        line = line.split()
        #������õĴ�������ȥ��š�������أ�
        tmp = []
        for i in range(len(line)):
            if not hasNum2(line[i]):
                tmp.append(line[i])
        data_list.append(tmp)
        line = f.readline()
        # ��˳���Ų��ظ��Ĵ���
        words = []
        for i in range(len(data_list)):
            for j in range(len(data_list[i])):
                if data_list[i][j] not in words:
                    words.append(data_list[i][j])

    #-------------------------------------

    f.close()
    # print(data_list)
    # ����ÿһ�еĴ��������в��ظ��Ĵ�����������
    return data_list,words,len(data_list)
```

### �ڼ���one-hot����
˼·����Ϊǰ������һ������Ԥ������������ֻ��Ҫ��һ������ѭ������ÿ�������ڵ�n���ı��Ĵ������Ϊ1,�Ϳ�����
Ϊ�˷����������one-hot����ÿһ��������ϲ���һ���ַ����ˣ���arrayҲ�ǿ��Եģ���������
```
def cal_one_hot(data,words,row):
    output = sys.stdout
    one_hot = open('one_hot','w')
    sys.stdout = one_hot

    #key code----------------------------------
    #result��¼��������ÿһ��Ϊһ��string
    result = []
    for i in range(row):
        string = ""
        for j in range(len(words)):
            if words[j] in data[i]:
                string = string + "1 "
            else:
                string = string + "0 "
        result.append(string)
    for i in range(len(result)):
        print(result[i])
    #----------------------------------------

    sys.stdout = output
    one_hot.close()
    print('done1')
```
### �ۼ���tf����
˼·����һ���ֵ�ͳ��ÿ���ı�������������Լ����ֵĴ������ܴ�����Ȼ��һ����ok�ˡ�
```
def cal_tf(data,words,row):
    output = sys.stdout
    tf = open('tf','w')
    sys.stdout = tf

    #Ϊ�˷�����һid��tf��������㣬����Ͱѽ������һ�����鷵��
    #key code------------------------------------
    array = [[0 for i in range(len(words))] for i in range(row)]
    #���ֵ���ÿһ���ı��������ֵĴ���
    d = dict()
    for i in range(row):
        sum = 0;
        string = ""
        d.clear()
        for j in range(len(data[i])):
            d[data[i][j]] = d.get(data[i][j],0)+1
            sum = sum + 1
        for j in range(len(words)):
            array[i][j] = float(d.get(words[j],0))/sum
            string = string +str(array[i][j])+' '
        print(string)
    #----------------------------------------------

    tf.close()
    sys.stdout = output
    print('done2')
    return array
```

### �ܼ���tf_idf����
˼·������ǰ���Ѿ����tf��������ֻ��Ҫ��һ������ѭ����Ȼ���ֵ��¼ÿ���ʳ�����ÿƪ���µĴ�������
```
def cal_tf_idf(data,words,row,tf):
    output = sys.stdout
    tf_idf = open('tf_idf','w')
    sys.stdout = tf_idf

#keycode--------------------------------------------
    #�ֵ�ͳ��ÿ���ʳ��ֵ�������
    d = dict()
    for i in range(len(words)):
        for j in range(row):
            if words[i] in data[j]:
                d[words[i]] = d.get(words[i],0)+1

    #���ù�ʽ����tf_idf����
    for i in range(row):
        string = ""
        for j in range(len(words)):
            tf_idf_ij = tf[i][j]*math.log(float(row)/d[words[j]])/math.log(2)
            string = string + str(tf_idf_ij) + ' '
        print(string)
        string = ""

  #---------------------------------------------------
    tf_idf.close()
    sys.stdout = output
    print('done3')
```

### ��one_hot����Ԫ��ϡ�裩����

˼·����һ���б��ϡ�����ǰ����ֱ�Ϊ����������Ч����������ÿһ����һ������Ч������ֵ���кš��кŵ��б�
</br>
ϡ���������������䣬����������λ�ö���ѭ���ҳ���
```
result = [];
  result.append(row)
  result.append(col)
  result.append(0)

  num = 0
  for i in range(row):
      for j in range(col):
          if not onehot[i][j] == 0:
              result.append([1,i,j])
              num = num + 1
  result[2] = num
```

### ��ϡ�����ӷ�
˼·���Ȱ�A�����Ƴ���ΪC��Ȼ����B����������û��λ����A��������ͬ�ģ�������ֵ���</br>
û������뵽C�����У���C������Ч��+1�����������кŶ�C��������
```
def plus(a,b):
    c = a
    for i in range(b[2]):
        flag = False
        for j in range(a[2]):
            #�������к���ͬ���������
            if a[3+j][1] == b[3+i][1] and a[3+j][2] == b[3+i][2]:
                c[3+j][0] = a[3+j][0] + b[3+i][0]
                flag = True
        #�µ���Ч�������뵽C����C��Ч��+1
        if not flag:
            c.append([b[i+3][0],b[i+3][1],b[i+3][2]])
            c[2] = c[2]+1
    #�������кŶ�C����
    c[3:len(c)] = c_part
    return c
    c_part = sorted(c[3:len(c)],key=lambda x:(x[1],x[2]))
```

## ����ʵ����չʾ
### ��one_hot����
������ǰ10���ı��Ľ����one_hot�����¼��������˳�����ı�һ�£���10�е�һ��"1"Ҳ�����ˡ�happi����ǰ���ظ��ĵ���
<img src = "img/002.png">
<img src = "img/003.png">
</br>������1��С���ӣ���һ�¼���case���ı�����û���ֹ���ȫ�����ֹ���������ȷ</br>
<img src="img/4.png">
<img src="img/5.png">

### ��tf����
������һ�����ӵ���֤����ȷ</br>
<img src="img/6.png"></br>
����������һ������</br>
<img src="img/7.png"></br>
������TA����������ǰ����</br>
<img src="img/8.png"></br>


### ��idf_tf����
ֻ�е����д����������¶����֣��űȽϺ�������֤����ͼ</br>
<img src="img/9.png" height="200"></br>
����һ�����,���Ҳ����Ԥ��</br>
<img src="img/10.png" width="320"></br>
��Ϊlog��2Ϊ�ף�Ūһ��4���ı���С���ݣ�һ�ֳ���2�Σ�һ�ֳ���1��</br>
����1�ε�Ϊ 0.5��log��4/1��= 1 </br>
����2�ε�Ϊ 0.5��log��4/2�� = 0.5 </br>
<img src="img/11.png" width="320"></br>
����ͼTa�ṩ���ݼ���ǰ����</br>
<img src="img/12.png"> </br>

### ��one_hotתϡ�����
��Ūһ����λ������һ��</br>
<img src="img/13.png" width="200"></br>
����ٸ�һ��С����</br>
<img src="img/14.png" width="200"></br>
������һ��������ǰ����</br>
<img src="img/15.png" width="200"></br>

### ����Ԫ��ӷ�
��һ������,�ӷ�������û����</br>
<img src="img/16.png" width="250">

## ˼����
Q��Ϊʲôid������һ���㷨��ĸҪ��1
> ��Ȼ���³��ֳ�0�������Ȼ��������һ�£���������һ�£���Ϊͳ�ƵĴ������ٶ����ֹ�һ�Σ���ô����ֳ�0�أ�Ȼ������һ�룬��һ�����оͳ����ˣ���Ϊ�ҵļ����Ǹ����ı�����ͳ�Ƶģ���������ı�û�д�����Ҳȥ�㣬Ȼ��ͳ��ֳ�0�ˡ�

Q��IDF��TF_IDF��ʲô���壿
>  IDF�������ļ�Ƶ�ʣ�ͳ�Ƶ���һ���ʳ����ڲ�ͬ���µĴ���������idf��Ļ�˵������ʳ��ֵ����²��࣬��˾������ֶȣ�����ʹ���������Ϊ���������������������Ȩ�أ���tf_idf��Ϊ�����һ��tf��������������������µ�Ƶ�ʳ����ȣ�������Ϊ���ú�idf���ƣ�tf�Ǻ���һ�������Ը��ı�����Ҫ�Ե�ָ�꣬����tf_idf�Գ����ĳ���ͬһ�������������֣���Ϊ���ĳ���һ�������Ͷ��ĳ���һ����������Ҫ����Ȼ�ǲ�һ���ģ����ֶ�Ҳ��Ȼ��һ������˸�����Ϊtf_idf��Ϊ����������������㡣

Q��Ϊʲô����Ԫ�����ϡ�����
> ��Ȼ����Ϊϡ�������˷ѿռ䣬��Ԫ���Ƿ����ڡ�ֱ�ۡ���ʡ�ռ�Ĵ��淽ʽ����
