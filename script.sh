# Experiment settings:
# 
#     dataset_name: "ConcreteCrack" or "AsphaltCrack"
#     num_of_clients: 1, 5, 10, 20
#     cfraction: 1, 0.75, 0.5, 0.25
#     epoch: 1, 5, 10
#     IID: 1, 0

# create logs folder and checkpoints folder
mkdir -p logs
mkdir -p checkpoints

# Run all experiments
for dataset_name in "ConcreteCrack" "AsphaltCrack"
do
    for num_of_clients in 1 5 10 20
    do
        for cfraction in 1 0.75 0.5 0.25
        do
            for epoch in 1 5 10
            do
                for IID in 1 0
                do
                    python server.py --dataset_name $dataset_name --num_of_clients $num_of_clients --cfraction $cfraction --epoch $epoch --IID $IID
                done
            done
        done
    done
done

