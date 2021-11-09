### 返回数据内容

#### res_bus

返回一个json数组，数组中每一个元素代表，每一次迭代中，每一个bus的状态

```json
[
    {
        "bus1":{             			
           static:{
               
           },
           dynamic:{
               
           }
        },
        "bus2":{
            // ...
        },
        // ...
        "bus30":{
        	// ...
    	}
	},
    {
        //...
    }
    
]
```

每一个bus对象中的内容。

| **Parameter** | **Datatype** | **Explanation**  | unit     | type    |
| ------------- | ------------ | ---------------- | -------- | ------- |
| zone | string |bus的归组 | - | static|
| vm_pu         | float        | 总线处的电压     | [p.u]    | dynamic |
| va_degree     | float        | 总线处电压角     | [degree] | dynamic |
| p_mw          | float        | 收敛后的有功需求 | [MW]     | dynamic |
| q_mvar        | float        | 收敛后的无功需求 | [Mvar]   | dynamic |

#### res_line

返回一个json数组，数组中每一个元素代表每一次迭代中，每一条线路的状态

```
[
    {
        "line1":{
        	"static":{
        	
        	},
        	"dynamic":{
        	
        	}
           // ..
        },
        "line2":{
            // ..
        },
        // ... 
        "line37":{
        	// ...
    	}
	},
	// ... 
    {
        //...
    }
    
]
```

每一个line对象中的内容。

| **Parameter**   | **Datatype** | **Explanation**                                        | unit | type |
| --------------- | ------------ | ------------------------------------------------------ | --------------- | --------------- |
|from_bus |string | 线路的起点“from” bus | - | static |
|to_bus|string|线路的终点”to” bus| - | static |
|length_km|string | 线路的长度 |km| static|
| max_i_ka| float| 最大发热电流 | kA| static|
|r_ohm_per_km|float| 线路阻抗|Ω/km|static|
|x_ohm_per_km| float|线路感抗|Ω/km|static|
|c_nf_per_km|float |线路容抗|nF/km|static|
|orientation | int | 线路电流方向，如果线路开启1表示从“from”总线去“to” 总线，2则相反， 0表示线路未开启。 | - | dynamic |
|vm_from_pu| float |在“from”总线处的电压| [p.u]  | dynamic |
|vm_to_pu| float |在“to”总线处的电压| [p.u]  | dynamic |
| p_from_mw       | float        | 在“from”总线处流入线路的有功功率 | [MW] | dynamic |
| q_from_mvar     | float        | 在“from”总线处流入线路的无功功率 | [MVar] | dynamic |
| p_to_mw         | float        | 在“to”总线处流入线路的有功功率 | [MW] | dynamic |
| q_to_mvar       | float        | 在“to”总线处流入线路的无功功率 | [MVar] | dynamic |
| pl_mw           | float        | 在线路上损耗的有功功率         | [MW] | dynamic |
| ql_mvar         | float        | 线路无功功率消耗  | [MVar] | dynamic |
| i_from_ka       | float        | 在“from”总线处流                       | [kA] | dynamic |
| i_to_ka       | float        | 在“to”总线处电流                       | [kA] | dynamic |
| i_ka |float| Maximum of i_from_ka and i_to_ka |[kA]|dynamic|



#### res_gen

返回一个json数组，数组中每一个元素代表每一次迭代中，每一个发电机的状态。

```json
[
    {
        "G1":{             			
            static:{
                
            },
            dynamic:{
                
            }
        },
        "G2":{
            // ... 
        },
        // ...
        "G6":{
        	// ...
    	}
	},
    {
        //...
    }
    
]
```

每一个gen对象中的内容。

| **Parameter** | **Datatype** | **Explanation**                                      | unit | type |
| ------------- | ------------ | ---------------------------------------------------- | ------------- | ------------- |
|bus | string | 发电机所在的bus名称 | - | static |
|is_black_start|boolean |是否是黑启动电机|-| static|
|max_q_mvar|float|发电机最大无功|   [MVar]       |static|
|min_q_mvar|float|发电机最小无功|[MVar] |static|
| is_open| boolean | 当前发电机开关状态| - | dynamic |
| p_mw          | float        | 发电机处的电压     | [MW] | dynamic |
| q_mvar        | float        | 发电机电压角 | [MVar] | dynamic |
| va_degree     | float        | 收敛后的有功功率                     | [degree] | dynamic |
| vm_pu         | float        | 收敛后的无功功率                       | [p.u] | dynamic |

#### res_load

返回一个json数组，数组中每一个元素代表每一次迭代中，每一个Load的状态。

```json
[
    {
        "L1":{             			
            static:{
                
            },
            dynamic:{
                
            }
        },
        "L2":{
            // ... 
        },
        // ...
        "L21":{
        	// ...
    	}
	},
    {
        //...
    }
    
]
```

每一个Load对象中的内容。

| **Parameter** | **Datatype** | **Explanation**                                              | unit | type |
| ------------- | ------------ | ------------------------------------------------------------ | ------------- | ------------- |
|bus| string | load所在bus名称| - | static |
|priority|int|负载开启优先级|-|staitc|
| p_steady_mw | float        | 负载完全开启后的有功功率 | [MW] | staitc |
| q_steady_mw | float        | 负载完全开启后的无功功率 | [MVar] | staitc |
|is_open| boolean |当前负载是否开启| - | dynamic |
| p_mw        | float        | 潮流收敛后，负载当前的有功功率 | [MW] | dynamic |
| q_mvar        | float        | 潮流收敛后，负载当前的有功功率 | [MVar] | dynamic |

