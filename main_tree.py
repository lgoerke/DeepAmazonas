from tree_test import test_Tree, Node
from matplotlib import pyplot as plt
import data_hdf5_tree as data
from data_hdf5_tree import Validation_splitter
from data_hdf5_tree import HDF_line_reader


LABELS = {'blow_down': 0,
          'bare_ground': 1,
          'conventional_mine': 2,
          'blooming': 3,
          'cultivation': 4,
          'artisinal_mine': 5,
          'haze': 6,
          'primary': 7,
          'slash_burn': 8,
          'habitation': 9,
          'clear': 10,
          'road': 11,
          'selective_logging': 12,
          'partly_cloudy': 13,
          'agriculture': 14,
          'water': 15,
          'cloudy': 16}
          

def add_class(names ):
    for name in names:
        LABELS[name] = len(LABELS.keys())


def invest_Tree():
    '''
     |       Little test tree... 
     |
    / \
    | |
    '''

    size = 64
    batch_size = 96
    nb_epoch = 1
    optimizer = 'adadelta'
    val_split = 0.2
    N_CLASSES = 17
    N_SAMPLES = 40479
    N_TEST = 61191
    test_batch_size = 9

    #labels = list(data.LABELS.keys())


    sp = Validation_splitter('/media/sebastian/7B4861FD6D0F6AA2/train.h5', val_split)
    train_reader = HDF_line_reader('/media/sebastian/7B4861FD6D0F6AA2/train.h5', load_rgb = False, img_size=size)
    test_reader = HDF_line_reader('/media/sebastian/7B4861FD6D0F6AA2/train.h5', load_rgb = False, img_size=size)
    
    new_cols = {'infrastructure': ['conventional_mine', 'artisinal_mine', 'slash_burn','bare_ground','habitation' ],
                        'forest_phenomena': ['blow_down', 'blooming', 'cultivation', 'slash_burn', 'selective_logging'] }
                        
    add_class(new_cols.keys())
    
    node_3_1_phenomena_investigator = Node('node_3_1_phenomena_investigator', [],
                    ['blow_down', 'blooming', 'slash_burn', 'selective_logging'],
                    ['forest_phenomena'],add_labels = {'forest_phenomena': ['blow_down', 'blooming', 'cultivation','slash_burn', 'selective_logging']},
                    clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)

    node_2_1_forestphenomena = Node('node_2_1_forestphenomena',
                    [node_3_1_phenomena_investigator],
                    ['forest_phenomena'],
                    ['primary'],
                    add_labels = {'forest_phenomena': ['blow_down', 'blooming', 'slash_burn', 'selective_logging']},clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)

    node_3_0_infra_investigator = Node('node_3_0_infra_investigator', [],
                    ['conventional_mine', 'artisinal_mine', 'road','slash_burn','bare_ground','habitation' ],['infrastructure'],
                    add_labels = {'infrastructure': ['conventional_mine', 'artisinal_mine', 'slash_burn','bare_ground','habitation' ]},clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)

    node_2_0_infrastructure = Node('node_2_0_infrastructure',
                    [node_3_0_infra_investigator],
                    ['infrastructure'],['haze', 'partly_cloudy', 'clear'],
                    add_labels = {'infrastructure': ['conventional_mine', 'artisinal_mine', 'slash_burn','bare_ground','habitation' ]},
                    clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)

    node_1_land = Node('node_1_land',
                    [node_2_0_infrastructure, node_2_1_forestphenomena],
                    ['agriculture', 'habitation', 'water', 'primary', 'cultivation', 'bare_ground'],
                    ['haze', 'partly_cloudy', 'clear'],clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)

    node_0_weather = Node('node_0_weather', [node_1_land], ['cloudy', 'haze', 'partly_cloudy', 'clear'],clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)

    return node_0_weather,sp

def weather_Tree():
    '''
    Tree that trains on all weather predictions apart.
    '''
    
    size = 64
    val_split = 0.3
    test_batch_size = 9

    #labels = list(data.LABELS.keys())


    sp = Validation_splitter('/media/sebastian/7B4861FD6D0F6AA2/train.h5', val_split)
    train_reader = HDF_line_reader('/media/sebastian/7B4861FD6D0F6AA2/train.h5', load_rgb = False, img_size=size)
    test_reader = HDF_line_reader('/media/sebastian/7B4861FD6D0F6AA2/train.h5', load_rgb = False, img_size=size)
    
    node_1_haze = Node('node_1_haze',
                    [],
                    ['blow_down','bare_ground','conventional_mine','blooming','cultivation',
          'artisinal_mine','primary','slash_burn','habitation','road','selective_logging',
          'agriculture','water'],
                    ['haze'],clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)
    
    node_2_clear = Node('node_2_clear',
                    [],
                    ['blow_down','bare_ground','conventional_mine','blooming','cultivation',
          'artisinal_mine','primary','slash_burn','habitation','road','selective_logging',
          'agriculture','water'],
                    ['clear'],clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)
                        
    node_3_partly_cloudy = Node('node_3_partly_cloudy',
                    [],
                    ['blow_down','bare_ground','conventional_mine','blooming','cultivation',
          'artisinal_mine','primary','slash_burn','habitation','road','selective_logging',
          'agriculture','water'],
                    ['partly_cloudy'],clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)
                                  
    
    node_0_weather = Node('node_0_weather', [node_1_haze,node_2_clear,node_3_partly_cloudy], ['cloudy', 'haze', 'partly_cloudy', 'clear'],clsfr = None,
                        validation_splitter = sp, train_reader = train_reader, test_reader = test_reader)

    return node_0_weather,sp    
    




n,sp = weather_Tree()
n.children[0].train()
sp.next_fold()
n.train_rec(True)

p_valid = n.apply_rec()
'''
print('validating')
p_valid = n.apply_rec()
#p_valid = p_valid[:len(sp.val_idx)]
idx = list(sp.val_idx)
idx.sort()
val_labels = reader.labels[idx]

loss = fbeta_score(val_labels, np.array(p_valid) > 0.2, beta=2, average='samples')

print('validation loss: {}'.format(loss))
'''
preds = []
for i in tqdm(range(p_valid.shape[0]), miniters=1000):
    a = p_valid.ix[[i]]
    a = a.apply(lambda x: x > thres_opt, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df = pd.DataFrame(np.zeros((N_TEST,2)), columns=['image_name','tags'])
df['image_name'] = test_reader.filenames
df['tags'] = preds

id = 0
df.to_csv('submission{}.csv'.format(id), index=False)