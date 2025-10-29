## DGL
This is the GitHub repository for 'DGL: Dynamic Global-Local Information Aggregation for Scalable VRP Generalization with Self-Improvement Learning'.

![image](image.png)

## DataSet
We use the same dataset as [INViT](https://github.com/Kasumigaoka-Utaha/INViT), Please prepare your dataset under the data directory in the project root directory first.

## How to run
**train**
~~~shell
cd CVRP
python train.py

cd ..
cd TSP
python train.py
~~~

**test**
~~~shell
cd CVRP
python test.py

cd ..
cd TSP
python test.py
~~~

⚠️ **Notice on Experimental Results (Updated: [2025/10/24])**

We have identified that the ELG configuration was unfair and have now corrected its results(pomo_size=50). \
![Experimental_Results](Experimental_Results.jpg) 

## Acknowledgments
We would like to thank the following repository, which is the baseline of our code:

https://github.com/Kasumigaoka-Utaha/INViT

https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD

## Citation
If you find our paper and code useful, please cite our paper:

```tex
@inproceedings{xiao2025dgl,
  title={{DGL}: Dynamic Global-Local Information Aggregation for Scalable {VRP} Generalization with Self-Improvement Learning},
  author={Xiao, Yubin and Wu, Yuesong and Cao, Rui and Wang, Di and Cao, Zhiguang and Zhao, Peng and Li, Yuanshu and Zhou, You and Jiang, Yuan},
  booktitle={Proceedings of International Joint Conference on Artificial Intelligence},
  pages={1-9},
  year={2025}
}
```


