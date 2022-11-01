local lib = import 'infotab_train.libsonnet';


local sent_selection_train = lib.trainer("sent_selection", "temp/seed/sent_selection/data/");

{
    steps: sent_selection_train
}