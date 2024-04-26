from trainer import Trainer
from options import myoption
import pandas
import os

tr = Trainer(myoption, device="cpu")

answer = input(f'Are you sure you want to {myoption.mode} the model? [y/n]')
if answer != 'y':
    exit()

if myoption.mode == 'train':
    tr.train()
elif myoption.mode == 'test':
    train_stats, val_stats, test_stats = tr.test()
    with open(os.path.join(tr.path_manager.models_path, 'logs.log'), 'a') as f:
        print('Train Stats:')
        print(train_stats, end='\n\n')
        print('Val Stats:')
        print(val_stats, end='\n\n')
        print('Test Stats:')
        print(test_stats)

        f.write('Train Stats: \n')
        f.write(str(train_stats) + '\n')
        f.write('Val Stats: \n')
        f.write(str(val_stats) + '\n')
        f.write('Test Stats: \n')
        f.write(str(test_stats) + '\n')
        f.write("##################################################################\n\n")

elif myoption.mode == 'feature':
    stats = tr.get_feature_importance()
    stats_df_dict = {'feature': []}
    i = 0
    for feature in stats:
        stats_df_dict['feature'].append(feature)
        for metric in stats[feature]:
            if metric not in stats_df_dict:
                stats_df_dict[metric] = []
            stats_df_dict[metric].append(stats[feature][metric])
        i += 1
        for key in stats_df_dict:
            if len(stats_df_dict[key]) < i:
                stats_df_dict[key].append(None)
    df = pandas.DataFrame(stats_df_dict)
    df = df.sort_values(by=df.columns[1], ascending=False)
    print(df)
    df.to_csv(f'./exp/{myoption.exp_name}/feature_importance.csv')
    
    
    
else:
    raise ValueError('Mode can only take thse values: [train, test, feature]')