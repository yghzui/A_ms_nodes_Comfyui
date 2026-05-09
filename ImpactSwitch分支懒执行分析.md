# ImpactSwitch 分支懒执行分析

## 1. 先说结论

`ImpactSwitch` 之所以能只执行被选中的那条线路，而忽略其他分支，核心并不是普通的 `if/else` 判断，而是依赖了 ComfyUI 的懒执行机制：

1. 把候选输入声明为 `lazy=True`
2. 通过 `check_lazy_status()` 告诉执行器当前真正需要哪个输入
3. ComfyUI 只为这个被点名的输入建立强依赖并继续调度
4. 没有被点名的分支不会成为当前节点的实际执行依赖，因此通常不会被执行

所以本质上是：

- `doit()` 负责“从已经需要的输入里取值”
- `check_lazy_status()` 负责“告诉 ComfyUI 这次到底该算哪个输入”


## 2. 节点映射关系

在 `ComfyUI-Impact-Pack/__init__.py` 中：

- `"ImpactSwitch": "Switch (Any)"` 只是显示名称
- 真正的类映射是 `"ImpactSwitch": GeneralSwitch`

也就是说，`ImpactSwitch` 实际对应的是 `modules/impact/util_nodes.py` 里的 `GeneralSwitch`。


## 3. 关键代码入口

### 3.1 动态输入被声明为懒输入

`GeneralSwitch.INPUT_TYPES()` 中最关键的是这一段思路：

```python
dyn_inputs = {
    "input1": (any_typ, {"lazy": True, "tooltip": "Any input. When connected, one more input slot is added."}),
}
```

这里的重点是 `{"lazy": True}`。

它的含义不是“这个输入可选”，而是：

- 这个输入可以先不立刻求值
- 当前节点可以先告诉执行器自己真正需要哪个输入
- 执行器再去补算那个输入对应的上游链路


### 3.2 `check_lazy_status()` 决定真正需要的输入

`GeneralSwitch` 的核心逻辑：

```python
def check_lazy_status(self, *args, **kwargs):
    selected_index = int(kwargs["select"])
    input_name = f"input{selected_index}"

    if input_name in kwargs:
        return [input_name]
    else:
        return []
```

假设：

- `select = 2`

那么这里就会返回：

```python
["input2"]
```

这个返回值的意义是：

- 当前节点此时只需要 `input2`
- 不需要 `input1`、`input3`、`input4` 等其他分支


### 3.3 `doit()` 只是返回被选中的那个值

执行函数本身很简单，思路相当于：

```python
selected_index = int(kwargs["select"])
input_name = f"input{selected_index}"
return kwargs[input_name], selected_label, selected_index
```

这里只是把已经确定好的目标输入取出来返回。

如果没有前面的懒执行机制，这里即使只返回 `input2`，也不代表 `input1`、`input3` 的上游没有提前执行。

因此：

- 真正控制“哪些分支会被执行”的，不是 `doit()`
- 而是 `lazy=True + check_lazy_status()`


## 4. ComfyUI 执行器是怎么配合的

ComfyUI 的执行器在发现某个节点实现了 `check_lazy_status()` 后，会先调用它。

执行流程可以概括为：

1. 收集当前节点输入
2. 检查节点是否实现 `check_lazy_status()`
3. 调用该函数，得到“当前真正需要的输入名列表”
4. 把这些输入转成当前节点的强依赖
5. 当前节点先进入 `PENDING`
6. 等这些被点名的输入算完，再回来执行当前节点

影响结果的关键点是：

- 被 `check_lazy_status()` 返回的输入，会成为当前节点必须等待的依赖
- 没被返回的输入，不会成为这次执行所需依赖

所以当 `ImpactSwitch` 只返回 `["input2"]` 时：

- `input2` 所在链路会继续执行
- 其他未被请求的分支通常不会被调度执行


## 5. 为什么未选中的分支会被“忽略”

这个“忽略”并不是说节点内部收到所有值后再手动丢弃，而是更早发生在依赖建立阶段。

可以这样理解：

### 普通非懒执行节点

普通节点通常是：

1. 先把所有输入都算好
2. 再进入节点函数
3. 在函数里决定返回哪个结果

这种情况下，即使最终只用了一个输入，其余分支也可能已经跑完了。


### `ImpactSwitch` 的懒执行节点

`ImpactSwitch` 的流程则是：

1. 先看 `select`
2. 再告诉 ComfyUI 只需要 `inputN`
3. ComfyUI 只去补算 `inputN` 对应的上游
4. 其他分支因为不是本次依赖，所以不进入实际执行链

所以它不是“执行后丢弃”，而是“执行前剪枝”或“执行时按需请求”。


## 6. `sel_mode` 的作用

`GeneralSwitch` 的输入里有一个：

```python
"sel_mode": ("BOOLEAN", {...})
```

这个参数不是直接在 `doit()` 或 `check_lazy_status()` 内部起作用，而是主要配合 `impact_server.py` 的 prompt 前处理逻辑。


### 6.1 `select_on_execution`

这是更接近默认懒执行的模式：

- 在真正执行阶段，根据 `select` 动态决定要哪个输入
- 再通过 `check_lazy_status()` 请求那个分支

特点：

- 选择发生在执行期
- 更依赖 ComfyUI 的 lazy evaluation 机制


### 6.2 `select_on_prompt`

这个模式下，Impact Pack 会在 prompt 进入执行器之前，预先处理节点输入：

- 先分析 `select`
- 找出应该保留的那个 `inputN`
- 把其他 `inputX` 从 prompt 里删掉

这样等真正执行时，未选中的分支连输入连接都已经被移除了。

因此这个模式更像：

- 执行前结构裁剪

而 `select_on_execution` 更像：

- 执行时按需请求


## 7. 两层机制的关系

`ImpactSwitch` 实际上可以理解为有两层分支控制：

### 第一层：Impact Pack 自己的 prompt 前处理

当 `sel_mode = select_on_prompt` 时：

- 先在 prompt 阶段把未选中的输入删除


### 第二层：ComfyUI 的 lazy evaluation

在执行阶段：

- 通过 `lazy=True`
- 再通过 `check_lazy_status()`
- 最终只请求真正需要的那个输入


## 8. 一句话理解

一句话概括：

`ImpactSwitch` 不是“先把所有分支都跑一遍再选一个输出”，而是“先根据选择值声明当前只需要哪一路输入，再让 ComfyUI 只去执行那一路的上游”。


## 9. 实际开发时可复用的设计思路

如果后续在 `A_my_nodes` 里也要实现类似“只执行被选中的分支”的节点，可以直接复用这个模式：

1. 把候选输入声明成 `lazy=True`
2. 提供 `check_lazy_status()`
3. 在 `check_lazy_status()` 中只返回当前真正需要的输入名
4. 在 `doit()` 里只读取被选中的输入并返回

这样做的好处是：

- 可以避免无效分支执行
- 可以减少上游节点的无意义计算
- 在复杂工作流里更容易控制执行链路


## 10. 注意事项

### 10.1 不要把 `doit()` 当成分支剪枝的核心

很多人看到 `doit()` 里只取 `inputN`，会误以为这就是“只执行这条分支”的原因。

实际上不是。

真正决定未选分支是否被执行的关键，是：

- 输入是否为 `lazy`
- `check_lazy_status()` 是否只返回需要的输入


### 10.2 `select_on_prompt` 与 `select_on_execution` 不是同一层逻辑

- `select_on_prompt` 是 prompt 预处理阶段改图结构
- `select_on_execution` 是执行阶段按需请求输入

两者效果相似，但发生时机不同。


### 10.3 动态输入名要和返回值一致

如果你的节点输入名是：

```python
input1, input2, input3
```

那么 `check_lazy_status()` 返回的也必须是：

```python
["input2"]
```

而不能返回别的别名，否则执行器无法正确识别。


## 11. 最终总结

`ImpactSwitch` 能忽略其他分支，核心原因有两点：

1. 输入槽被声明为 `lazy=True`
2. `check_lazy_status()` 只向 ComfyUI 请求被选中的那个输入

补充上：

- `select_on_prompt` 可以在执行前直接裁掉未选中的连接
- `doit()` 只负责最终返回被选中的值，不负责上游执行裁剪

因此这个节点的本质是：

- 先决定需要哪个输入
- 再只执行那个输入对应的链路

而不是：

- 把所有分支都执行完后再做结果筛选
