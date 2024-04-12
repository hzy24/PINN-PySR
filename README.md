
## PySRRegressor 参数
- **model_selection** : str
    当从每个复杂度级别的最佳表达式列表中选择最终表达式时的模型选择标准。
    可选值为`'accuracy'`、`'best'`或`'score'`。默认值为`'best'`。
    - `'accuracy'`：选择具有最低损失（最高准确度）的候选模型。
    - `'score'`：选择具有最高分数的候选模型。分数定义为对数损失的导数的负数。
    - `'best'`：在损失好于至少1.5倍最准确模型的表达式中选择分数最高的候选模型。

- **binary_operators** : list[str]
    用于搜索的二元运算符列表。详见[运算符页面](https://astroautomata.com/PySR/operators/)。
    默认为`["+", "-", "*", "/"]`。

- **unary_operators** : list[str]
    只接受单个标量输入的运算符。
    例如，`"cos"` 或 `"exp"`。
    默认值为`None`。

- **niterations** : int
    运行算法的迭代次数。每次迭代结束时，最佳方程式将被打印并在种群之间迁移。
    默认值为`40`。

- **populations** : int
    运行的种群数量。
    默认值为`15`。

- **population_size** : int
    每个种群中的个体数量。
    默认值为`33`。

- **max_evals** : int
    将表达式评估次数限制在此数字。
    默认值为`None`。

- **maxsize** : int
    方程的最大复杂度。
    默认值为`20`。

- **maxdepth** : int
    方程的最大深度。可以同时使用`maxsize`和`maxdepth`。`maxdepth`默认不使用。
    默认值为`None`。

- **warmup_maxsize_by** : float
    是否从一个小数字慢慢增加到maxsize（如果大于0）。如果大于0，表示在训练时间的这个比例时，当前的maxsize将达到用户传递的maxsize。
    默认值为`0.0`。

- **timeout_in_seconds** : float
    搜索达到多少秒后提前返回。
    默认值为`None`。

- **constraints** : dict[str, int | tuple[int,int]]
    字典，int（一元）或2元组（二元），强制执行运算符参数的最大大小限制。例如，`'pow': (-1, 1)`表示幂律可以有任何复杂度的左参数，但右参数只能有1的复杂度。使用此参数可以强制获得更易解释的解决方案。
    默认值为`None`。

- **nested_constraints** : dict[str, dict]
    指定运算符可以嵌套的次数。例如，`{"sin": {"cos": 0}}, "cos": {"cos": 2}}`指定`cos`不能在`sin`中出现，但`sin`可以无限次嵌套自身。第二项指定`cos`可以在`cos`中嵌套多达2次。
    默认值为`None`。

- **elementwise_loss** : str
    指定元素级损失函数的Julia代码字符串。可以是LossFunctions.jl中的损失函数，也可以是您自己编写的函数。自定义编写的损失函数示例包括：`myloss(x, y) = abs(x-y)`（非加权）或`myloss(x, y, w) = w*abs(x-y)`（加权）。
    默认值为`"L2DistLoss()"`。

- **loss function** : str
    或者，您可以指定完整的目标函数作为Julia代码片段，包括任何类型的自定义评估（包括之前的符号操作），以及任何类型的损失函数或正则化。SymbolicRegression.jl中使用的默认`loss_function`大致等同于：
    ```julia
    function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        prediction, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        }
        return sum((prediction .- dataset.y) .^ 2) / dataset.n
    end
    ```
    默认值为`None`。

- **complexity_of_operators** : dict[str, float]
    如果您想为运算符使用其他复杂度，请在这里指定。例如，`{"sin": 2, "+": 1}`为使用`sin`运算符的每次使用指定2的复杂度，为使用`+`运算符的每次使用指定1的复杂度（这是默认值）。您可以为复杂度指定实数，并且在计算后将树的总复杂度四舍五入到最接近的整数。
    默认值为`None`。

- **complexity_of_constants** : float
    常数的复杂度。默认值为`1`。

- **complexity_of_variables** : float
    变量的复杂度。默认值为`1`。

- **parsimony** : float
    对复杂度的惩罚的乘法因子。
    默认值为`0.0032`。

- **dimensional_constraint_penalty** : float
    如果表达式的维度分析失败，则添加罚分。默认值为`1000.0`。

- **use_frequency** : bool
    是否测量复杂度的频率，并使用它而不是简单地探索方程空间。将自然发现所有复杂度的方程。
    默认值为`True`。

- **use_frequency_in_tournament** : bool
    是否在锦标赛中使用上述频率，而不仅仅是模拟退火。
    默认值为`True`。

- **adaptive_parsimony_scaling** : float
    如果使用自适应节俭策略（`use_frequency`和`use_frequency_in_tournament`），这是多少权重（指数地）贡献。如果发现搜索仅优化最复杂的表达式而简单表达式停滞不前，您应该增加此值。
    默认值为`20.0`。

- **alpha** : float
    模拟退火的初始温度（需要`annealing`为`True`）。
    默认值为`0.1`。

- **annealing** : bool
    是否使用退火。默认为`False`。

- **early_stop_condition** : float | str
    如果达到此损失，则提前停止搜索。您还可以传递一个包含Julia函数的字符串，该函数将损失和复杂度作为输入，例如：`"f(loss, complexity) = (loss < 0.1) && (complexity < 10)"`。
    默认值为`None`。

- **ncycles_per_iteration** : int
    每次迭代中每10个样本的种群进行的总变异次数。
    默认值为`550`。

- **fraction_replaced** : float
    用来自其他种群迁移的方程替换种群的多少。
    默认值为`0.000364`。

- **fraction_replaced_hof** : float
    用来自名人堂的方程替换种群的多少。默认值为`0.035`。

- **weight_add_node** : float
    变异添加节点的相对可能性。
    默认值为`0.79`。

- **weight_insert_node** : float
    变异插入节点的相对可能性。
    默认值为`5.1`。

- **weight_delete_node** : float
    变异删除节点的相对可能性。
    默认值为`1.7`。

- **weight_do_nothing** : float
    变异不改变个体的相对可能性。
    默认值为`0.21`。

- **weight_mutate_constant** : float
    变异稍微随机改变常数的相对可能性。
    默认值为`0.048`。

- **weight_mutate_operator** : float
    变异交换运算符的相对可能性。
    默认值为`0.47`。

- **weight_swap_operands** : float
    在二元运算符中交换操作数的相对可能性。
    默认值为`0.1`。

- **weight_randomize** : float
    变异完全删除然后随机生成方程的相对可能性。
    默认值为`0.00023`。

- **weight_simplify** : float
    变异通过计算简化常数部分的相对可能性。
    默认值为`0.0020`。

- **weight_optimize** : float
    常数优化也可以作为变异进行，除了由`optimize_probability`控制的正常策略，每次迭代都会发生。如果您想使用大量的`ncycles_per_iteration`，并且可能不会经常优化，使用它作为变异是有用的。
    默认值为`0.0`。

- **crossover_probability** : float
    交叉类型遗传操作的绝对概率，而不是变异。
    默认值为`0.066`。

- **skip_mutation_failures** : bool
    是否跳过变异和交叉失败，而不是简单地重新采样当前成员。
    默认值为`True`。

- **migration** : bool
    是否进行迁移。默认为`True`。

- **hof_migration** : bool
    是否让名人堂进行迁移。默认为`True`。

- **topn** : int
    每个种群迁移的顶级个体数量。
    默认值为`12`。

- **should_simplify** : bool
    是否在搜索中使用代数简化。请注意，只实现了一些简单的规则。默认为`True`。

- **should_optimize_constants** : bool
    是否在每次迭代结束时对常数进行数值优化（Nelder-Mead/Newton）。
    默认值为`True`。

- **optimizer_algorithm** : str
    用于优化常数的优化方案。目前可以是`NelderMead`或`BFGS`。
    默认值为`"BFGS"`。

- **optimizer_nrestarts** : int
    常数优化过程中使用不同初始条件重新启动的次数。
    默认值为`2`。

- **optimize_probability** : float
    在一次进化算法的迭代中优化常数的概率。
    默认值为`0.14`。

- **optimizer_iterations** : int
    常数优化器可以进行的迭代次数。
    默认值为`8`。

- **perturbation_factor** : float
    常数被扰动的最大因子为（perturbation_factor*T + 1）。可能会被这个乘以或除以。
    默认值为`0.076`。

- **tournament_selection_n** : int
    在每次锦标赛中考虑的表达式数量。
    默认值为`10`。

- **tournament_selection_p** : float
    在每次锦标赛中选择最佳表达式的概率。对于其他按损失排序的表达式，概率将衰减为p*(1-p)^n。
    默认值为`0.86`。

- **procs** : int
    进程数（=运行的种群数）。
    默认值为`cpu_count()`。

- **multithreading** : bool
    是否使用多线程而不是分布式后端。设置procs=0将关闭多线程和分布式。默认为`True`。

- **cluster_manager** : str
    对于分布式计算，这将设置作业队列系统。设置为
    其中之一"slurm"、"pbs"、"lsf"、"sge"、"qrsh"、"scyld"或"htc"。如果设置为这些之一，PySR将在分布式模式下运行，并使用`procs`来确定启动多少个进程。
    默认值为`None`。

- **heap_size_hint_in_bytes** : int
    对于多进程，这设置了新Julia进程的`--heap-size-hint`参数。在使用多节点分布式计算时可以配置，为每个进程提供一个关于它们可以使用多少内存的提示，以便在积极的垃圾回收之前。

- **batching** : bool
    是否在进化过程中使用小批量比较种群成员。仍然使用完整数据集与名人堂进行比较。默认为`False`。

- **batch_size** : int
    如果进行批处理，使用的数据量。默认为`50`。

- **fast_cycle** : bool
    批处理种群样本。这是一种与常规进化略有不同的算法，但循环速度快15%。算法上可能效率较低。
    默认为`False`。

- **turbo** : bool
    （实验性）是否使用LoopVectorization.jl加速搜索评估。某些运算符可能不受支持。不支持16位精度浮点数。
    默认为`False`。

- **bumper** : bool
    （实验性）是否使用Bumper.jl加速搜索评估。不支持16位精度浮点数。
    默认为`False`。

- **precision** : int
    使用数据的精度。默认为`32`（float32），但您也可以选择`64`或`16`，分别为您提供64位或16位的浮点精度。如果您传递复杂数据，将使用相应的复杂精度（即，`64`对应complex128，`32`对应complex64）。
    默认为`32`。

- **enable_autodiff** : bool
    是否为自动微分创建运算符的导数版本。这仅在您希望在自定义损失函数中计算表达式的梯度时需要。
    默认为`False`。

- **random_state** : int, Numpy RandomState实例或None
    传递一个int以在多个函数调用之间获得可重现的结果。
    默认为`None`。

- **deterministic** : bool
    使PySR搜索每次运行时给出相同的结果。要使用此功能，您必须关闭并行性（使用`procs`=0, `multithreading`=False），并将`random_state`设置为固定种子。
    默认为`False`。

- **warm_start** : bool
    告诉fit从上次调用fit结束的地方继续。如果为false，每次调用fit都将是全新的，覆盖之前的结果。
    默认为`False`。

- **verbosity** : int
    使用的详细程度级别。0表示最小打印语句。
    默认为`1`。

- **update_verbosity** : int
    使用的包更新详细程度级别。如果未给出，将采用`verbosity`的值。
    默认为`None`。

- **print_precision** : int
    打印浮点数时要显示的有效数字位数。默认为`5`。

- **progress** : bool
    是否使用进度条而不是打印到stdout。
    默认为`True`。

- **equation_file** : str
    保存文件的位置（.csv扩展名）。
    默认为`None`。

- **temp_equation_file** : bool
    是否将名人堂文件放在临时目录中。然后通过`delete_tempfiles`参数控制删除。
    默认为`False`。

- **tempdir** : str
    临时文件的目录。默认为`None`。

- **delete_tempfiles** : bool
    是否在完成后删除临时文件。
    默认为`True`。

- **update** : bool
    是否在调用`fit`时自动更新Julia包。您应确保PySR本身已经是最新的，因为打包的Julia包可能不一定包括所有更新的依赖项。
    默认为`False`。

- **output_jax_format** : bool
    是否在输出中创建一个'jax_format'列，其中包含可调用的jax函数和默认参数的jax数组。
    默认为`False`。

- **output_torch_format** : bool
    是否在输出中创建一个'torch_format'列，其中包含一个带有可训练参数的torch模块。
    默认为`False`。

- **extra_sympy_mappings** : dict[str, Callable]
    提供自定义`binary_operators`或`unary_operators`在julia字符串中定义到同一运算符在sympy中定义的映射。例如，如果`unary_operators=["inv(x)=1/x"]`，那么为了将拟合模型导出到sympy，`extra_sympy_mappings`将是`{"inv": lambda x: 1/x}`。
    默认为`None`。

- **extra_jax_mappings** : dict[Callable, str]
    与`extra_sympy_mappings`类似，但用于模型导出到jax。字典将sympy函数映射到jax函数。例如：`extra_jax_mappings={sympy.sin: "jnp.sin"}`将`sympy.sin`函数映射到等价的jax表达式`jnp.sin`。
    默认为`None`。

- **extra_torch_mappings** : dict[Callable, Callable]
    与`extra_jax_mappings`相同，但用于模型导出到pytorch。请注意，字典键应为可调用的pytorch表达式。例如：`extra_torch_mappings={sympy.sin: torch.sin}`。
    默认为`None`。

- **denoise** : bool
    是否在输入到PySR之前使用高斯过程去噪数据。可以帮助PySR拟合噪声数据。
    默认为`False`。

- **select_k_features** : int
    是否在Python中使用随机森林进行特征选择，然后传递给符号回归代码。`None`表示不进行特征选择；一个int表示选择那么多特征。
    默认为`None`。

### 属性
- **equations_** : pandas.DataFrame | list[pandas.DataFrame]
    包含模型拟合结果的处理后的DataFrame。

- **n_features_in_** : int
    在`fit`期间看到的特征数量。

- **feature_names_in_** : ndarray of shape (`n_features_in_`,)
    在`fit`期间看到的具有全部字符串特征名称的`X`的特征名称。仅在定义时可用。

- **display_feature_names_in_** : ndarray of shape (`n_features_in_`,)
    仅在打印时使用的特征的漂亮名称。

- **X_units_** : list[str] of length n_features
    训练数据集`X`中每个变量的单位。

- **y_units_** : str | list[str] of length n_out
    训练数据集`y`中每个变量的单位。

- **nout_** : int
    输出维度的数量。

- **selection_mask_** : list[int] of length `select_k_features`
    当设置`select_k_features`时，选定输入特征的索引列表。

- **tempdir_** : Path
    临时方程文件目录的路径。

- **equation_file_** : str
    由julia后端产生的输出方程文件名。

- **julia_state_stream_** : ndarray
    序列化的julia SymbolicRegression.jl后端状态（拟合后），存储为uint8数组，由Julia的Serialization.serialize函数生成。

- **julia_state_**
    反序列化的julia状态。

- **julia_options_stream_** : ndarray
    序列化

的julia选项，存储为uint8数组。

- **julia_options_**
    反序列化的julia选项。

- **equation_file_contents_** : list[pandas.DataFrame]
    Julia后端输出的方程文件的内容。

- **show_pickle_warnings_** : bool
    是否显示有关哪些属性可以被pickle的警告。




# 实例代码
```python
from pysr import PySRRegressor
model = PySRRegressor(
    procs = 4,  # 使用4个进程并行运行，以加快符号回归的计算速度。
    populations = 8,  # 设定总的种群数为8，即每个核心运行2个种群，保证一直有任务在运行。
    population_size = 50,  # 每个种群的大小为50，这个大小决定了每一代中的公式数量。
    ncycles_per_iteration = 500,  # 每次迭代之间的生成次数，用于控制种群的迁移频率。
    niterations = 10000000,  # 设定迭代次数为10000000，相当于“无限”运行，除非达到停止条件。
    early_stop_condition = (
        "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
        # 如果找到一个既简单又准确（损失小于1e-6，复杂度小于10）的公式，则提前停止。
    ),
    timeout_in_seconds = 60 * 60 * 24,  # 设置超时时间为24小时，作为另一种停止条件。
    maxsize = 50,  # 允许公式的最大复杂度为50，以允许更复杂的模型。
    maxdepth = 10,  # 限制公式的最大嵌套深度为10，以避免过度嵌套。
    binary_operators = ["*", "+", "-", "/"],  # 指定可用的二元运算符。
    unary_operators = ["square", "cube", "exp", "cos2(x)=cos(x)^2"],  # 指定可用的一元运算符。
    constraints = {
        "/": (-1, 9),
        "square": 9,
        "cube": 9,
        "exp": 9,
        # 限制特定运算符中参数的复杂度。例如，"/"操作的分母的最大复杂度为9。
    },
    nested_constraints = {
        "square": {"square": 1, "cube": 1, "exp": 0},
        "cube": {"square": 1, "cube": 1, "exp": 0},
        "exp": {"square": 1, "cube": 1, "exp": 0},
        # 对运算符嵌套施加限制。例如，不允许"square(exp(x))"。
    },
    complexity_of_operators = {"/": 2, "exp": 3},  # 自定义特定运算符的复杂度。
    complexity_of_constants = 2,  # 将常数的复杂度设置高于变量，以减少常数的使用。
    select_k_features = 4,  # 仅在最重要的4个特征上训练模型。
    progress = True,  # 显示训练进度。如果输出到文件，可以设置为False。
    weight_randomize = 0.1,  # 更频繁地随机化树的权重，以增加搜索空间的多样性。
    cluster_manager = None,  # 可以设置为如"slurm"来在slurm集群上运行，仅需从头节点启动脚本。
    precision = 64,  # 使用更高的计算精度。
    warm_start = True,  # 从上次停止的地方继续，利用之前的运算结果。
    bumper = True,  # 启用更快的评估方法（实验性质）。
    extra_sympy_mappings = {"cos2": lambda x: sympy.cos(x)**2},  # 定义额外的SymPy到Python函数的映射。   
    extra_torch_mappings = {sympy.cos: torch.cos},  # 为PyTorch定义自定义操作符，此行为注释因为cos已定义。
    extra_jax_mappings = {sympy.cos: "jnp.cos"},  # 为JAX定义自定义操作符，通过字符串指定。
)
```
