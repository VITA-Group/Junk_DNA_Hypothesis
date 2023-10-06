# GPU=0
# bash multi_lingual/omp_after/omp_after_2to2.sh $GPU 0.2 1
# bash multi_lingual/omp_after/omp_after_2to2.sh $GPU 0.36 2
# bash multi_lingual/omp_after/omp_after_2to2.sh $GPU 0.488 3
# bash multi_lingual/omp_after/omp_after_2to2.sh $GPU 0.590 4
# bash multi_lingual/omp_after/omp_after_2to2.sh $GPU 0.672 5
# bash multi_lingual/omp_after/omp_after_2to2.sh $GPU 0.738 6
# bash multi_lingual/omp_after/omp_after_2to2.sh $GPU 0.791 7
# bash multi_lingual/omp_after/omp_after_2to2.sh $GPU 0.8325 8
# bash multi_lingual/omp_after/omp_after_2to2.sh $GPU 0.866 9
# bash multi_lingual/omp_after/omp_after_2to2.sh $GPU 0.893 10
# GPU=0
# bash multi_lingual/omp_after/omp_after_5to5.sh $GPU 0.2 1
# bash multi_lingual/omp_after/omp_after_5to5.sh $GPU 0.36 2
# bash multi_lingual/omp_after/omp_after_5to5.sh $GPU 0.488 3
# bash multi_lingual/omp_after/omp_after_5to5.sh $GPU 0.590 4
# bash multi_lingual/omp_after/omp_after_5to5.sh $GPU 0.672 5
# bash multi_lingual/omp_after/omp_after_5to5.sh $GPU 0.738 6
# bash multi_lingual/omp_after/omp_after_5to5.sh $GPU 0.791 7
# bash multi_lingual/omp_after/omp_after_5to5.sh $GPU 0.8325 8
# bash multi_lingual/omp_after/omp_after_5to5.sh $GPU 0.866 9
# bash multi_lingual/omp_after/omp_after_5to5.sh $GPU 0.893 10
# GPU=0
# bash multi_lingual/omp_after/omp_after_10to1.sh $GPU 0.2 1
# bash multi_lingual/omp_after/omp_after_10to1.sh $GPU 0.36 2
# bash multi_lingual/omp_after/omp_after_10to1.sh $GPU 0.488 3
# bash multi_lingual/omp_after/omp_after_10to1.sh $GPU 0.590 4
# bash multi_lingual/omp_after/omp_after_10to1.sh $GPU 0.672 5
# bash multi_lingual/omp_after/omp_after_10to1.sh $GPU 0.738 6
# bash multi_lingual/omp_after/omp_after_10to1.sh $GPU 0.791 7
# bash multi_lingual/omp_after/omp_after_10to1.sh $GPU 0.8325 8
# bash multi_lingual/omp_after/omp_after_10to1.sh $GPU 0.866 9
# bash multi_lingual/omp_after/omp_after_10to1.sh $GPU 0.893 10
# GPU=0
# bash multi_lingual/omp_after/omp_after_10to10.sh $GPU 0.2 1
# bash multi_lingual/omp_after/omp_after_10to10.sh $GPU 0.36 2
# bash multi_lingual/omp_after/omp_after_10to10.sh $GPU 0.488 3
# bash multi_lingual/omp_after/omp_after_10to10.sh $GPU 0.590 4
# bash multi_lingual/omp_after/omp_after_10to10.sh $GPU 0.672 5
# bash multi_lingual/omp_after/omp_after_10to10.sh $GPU 0.738 6
# bash multi_lingual/omp_after/omp_after_10to10.sh $GPU 0.791 7
# bash multi_lingual/omp_after/omp_after_10to10.sh $GPU 0.8325 8
# bash multi_lingual/omp_after/omp_after_10to10.sh $GPU 0.866 9
# bash multi_lingual/omp_after/omp_after_10to10.sh $GPU 0.893 10


nohup bash multi_lingual/omp_after/ex1.sh 0 > log_omp_after_ex1.out 2>&1 &
nohup bash multi_lingual/omp_after/ex2.sh 1 > log_omp_after_ex2.out 2>&1 &
nohup bash multi_lingual/omp_after/ex3.sh 2 > log_omp_after_ex3.out 2>&1 &
nohup bash multi_lingual/omp_after/ex4.sh 3 > log_omp_after_ex4.out 2>&1 &




