for i in {1..10}
do

    python exp_mag.py --dataname music --dataset itunes-amazon --no $i 
    python exp_mag.py --dataname dmusic --dataset itunes-amazon --no $i 

    python exp_mag.py --dataname citation --dataset dblp-scholar --no $i 
    python exp_mag.py --dataname dcitation --dataset dblp-scholar --no $i 

    python exp_mag.py --dataname citeacm --dataset dblp-acm --no $i 
    python exp_mag.py --dataname dciteacm --dataset dblp-acm --no $i 

done