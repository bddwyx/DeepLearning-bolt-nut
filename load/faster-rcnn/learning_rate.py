#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler
from paddle.fluid.layers import control_flow


def exponential_with_warmup_decay(learning_rate, boundaries, values,
                                  warmup_iter, warmup_factor):
    '''
    权重衰减系数为0.0001，前500轮学习率从0.00333线性增加至0.01。在120000，160000轮时使用0.1,0.01乘子进行学习率衰减，最大训练180000轮。
    '''
                                      
    global_step = lr_scheduler._decay_step_counter()    #当前iter数

    lr = fluid.layers.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")

    warmup_iter_var = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=float(warmup_iter), force_cpu=True)

    with control_flow.Switch() as switch:
        with switch.case(global_step < warmup_iter_var):   # iter数小于warmup_iter数，对learning_rate做如下调整，递增操作
            alpha = global_step / warmup_iter_var
            factor = warmup_factor * (1 - alpha) + alpha
            decayed_lr = learning_rate * factor
            fluid.layers.assign(decayed_lr, lr)

        for i in range(len(boundaries)):
            boundary_val = fluid.layers.fill_constant(    #当前iter数超过warmup_iter数时，当iter数小于边界步数时，将此时的对应的value_var的值赋值给lr
                shape=[1],
                dtype='float32',
                value=float(boundaries[i]),
                force_cpu=True)
            value_var = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=float(values[i]))
            with switch.case(global_step < boundary_val):
                fluid.layers.assign(value_var, lr)

        last_value_var = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=float(values[len(values) - 1]))
        with switch.default():#当前iter数超过最大边界步数时，将last_value_var的值赋值给lr
            fluid.layers.assign(last_value_var, lr)

    return lr
