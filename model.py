import tensorflow as tf
import tensorflow.keras.layers as tfkl

class TCN_Block(tf.keras.Model):

    def __init__(self,num_filters, k, d, last=False, first=False):
        super(TCN_Block, self).__init__()
        self.first = first
        self.last = last
        
        self.conv1 = tfkl.Conv1D(filters=num_filters,kernel_size=k,dilation_rate=d, padding='causal')
        if last:
            self.conv2 = tfkl.Conv1D(filters=1,kernel_size=k,dilation_rate=d, padding='causal')
        else:
            self.conv2 = tfkl.Conv1D(filters=num_filters,kernel_size=k,dilation_rate=d, padding='causal')

        if first:
            self.short_conv = tfkl.Conv1D(filters=num_filters,kernel_size=1)
        elif last:
            self.short_conv = tfkl.Conv1D(filters=1,kernel_size=1)

        self.relu1 = tfkl.ReLU()
        self.relu2 = tfkl.ReLU()


    def call(self, x):

        h = self.conv1(x)
        h = self.relu1(h)
        h = self.conv2(h)

        # shortcut
        if self.first or self.last:
            h = h + self.short_conv(x)
        else:
            h = h + x

        # last activation
        if self.last:
            out = h
        else:
            out = self.relu2(h)

        return out


class TCN(tf.keras.Model):

    def __init__(self, num_layers, num_filters, kernel_size, dilation_base):
        super(TCN, self).__init__()

        self.seq = tf.keras.Sequential()

        for i in range(num_layers):
            if i==0:
                self.seq.add(TCN_Block(num_filters, kernel_size, dilation_base**i, first=True))
            elif i==num_layers-1:
                self.seq.add(TCN_Block(num_filters, kernel_size, dilation_base**i, last=True))
            else:
                self.seq.add(TCN_Block(num_filters, kernel_size, dilation_base**i))
    
    def call(self, x):
        
        out = self.seq(x)
        
        return out


class NBEATS_Block(tf.keras.Model):

    def __init__(self, width, forecast_H, lookback_H):
        super(NBEATS_Block, self).__init__()
        self.fc1 = tfkl.Dense(units=width,activation='relu',use_bias=True)
        self.fc2 = tfkl.Dense(units=width,activation='relu',use_bias=True)
        self.fc3 = tfkl.Dense(units=width,activation='relu',use_bias=True)
        self.fc4 = tfkl.Dense(units=width,activation='relu',use_bias=True)

        self.b_linear = tfkl.Dense(units=width,activation=None,use_bias=False)
        self.f_linear = tfkl.Dense(units=width,activation=None,use_bias=False)

        self.g_b = tfkl.Dense(units=lookback_H,activation=None,use_bias=True)

        self.g_f = tfkl.Dense(units=forecast_H,activation=None,use_bias=True)

    def call(self, x):

        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        h4 = self.fc4(h3)

        theta_b = self.b_linear(h4)
        theta_f = self.f_linear(h4)

        x_hat = self.g_b(theta_b)
        y_hat = self.g_f(theta_f)

        return x_hat, y_hat


class NBEATS_Stack(tf.keras.Model):

    def __init__(self, blocks, width, forecast_H, lookback_H):
        super(NBEATS_Stack, self).__init__()
        self.blocks = blocks

        for idx in range(self.blocks):
            setattr(self,'block'+str(idx), NBEATS_Block(width, forecast_H, lookback_H))

    def call(self, x):
        
        x_hat, y_hat = getattr(self,'block'+str(0))(x)
        x_new = x - x_hat
        y_stack = y_hat
        
        for idx in range(1,self.blocks):
            x_hat, y_hat = getattr(self,'block'+str(idx))(x_new)
            x_new -= x_hat
            y_stack += y_hat

        return x_new, y_stack


class NBEATS(tf.keras.Model):

    def __init__(self, stacks, blocks, width, forecast_H, lookback_H):
        super(NBEATS, self).__init__()
        self.stacks = stacks

        for idx in range(self.stacks):
            setattr(self,'stack'+str(idx), NBEATS_Stack(blocks, width, forecast_H, lookback_H))

    def call(self, x):
        
        x_stack, y_stack = getattr(self,'stack'+str(0))(x)
        y_total = y_stack
        
        for idx in range(1,self.stacks):
            x_stack, y_stack = getattr(self,'stack'+str(idx))(x_stack)
            y_total += y_stack
        
        return y_total