# concerning   **TRM & VIT**

## TRM

![细节拉满，全网最详细的Transformer介绍（含大量插图）！](https://picx.zhimg.com/70/v2-b158466761a3f3753bec3aba2b7e77d8_1440w.image?source=172ae18b&biz_tag=Post)

transformer的**自回归特征**：即前一时刻的输出会成为后一时刻的输入。

![image-20250623104549533](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623104549533.png)

输入=embedding+位置嵌入 。
与RNN 根据时间线展开不同，TRM增加了速度但忽略了序列关系。

![image-20250623105538878](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623105538878.png)

QKV，把QK的相似度作为V的权重，分别为query、key与value。

![image-20250623151344670](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623151344670.png)

每个注意力attention头里输入X进行的系列运算

![image-20250623151553452](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623151553452.png)

softmax具体操作：即取自然对数为底后进行归一化运算

![image-20250623153526887](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623153526887.png)

解码器中masked-attention，为了说明前面输出的内容不受后面输出内容的影响，把后面的qk内积值设置为绝对值很大的负值以削弱其对特征（即需要关注的位置）的影响。

![image-20250623153946727](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623153946727.png)

解码器中的cross-attention“与self-attention区分，采用的三元组是编码器的kv与解码器的q。

![image-20250623105855694](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623105855694.png)

![image-20250623152238295](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623152238295.png)

使用的是同一套矩阵Wq，Wk，Wv

![image-20250623105955139](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623105955139.png)

![image-20250623111329766](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623111329766.png)

这样设计就不会梯度为0 

前馈神经网络中的feed forward是全连接层。add&norm模块，即分别进行一个残差操作（输入+输出）与一个标准化操作，即所有特征减去一个均值除以一个标准差。 

![image-20250623154337706](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623154337706.png)

最后经过线性方程linear（x）=Wx+b处理并softmax后得到各个概率矩阵，每个矩阵代表出现中文字的概率大小，将数字最大的位置置为1，其他位置置为0，这样就会得到如上所示的onehot独热编码。

***

## VIT

将图像转化为Embedding序列的两种实现方式：

![image-20250623155931265](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623155931265.png)

对位置编码的尝试，以9个patch为例：

1-D即生成9个特征维度一样为1024的可学习向量；而2-D即生成代表行的3个向量，3个代表列的，维度都为特征维度的一半即512，通过行列拼接可以得到1024个位置编码。 

![image-20250623160731347](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623160731347.png)

![image-20250623160854970](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20250623160854970.png)

