def pprint_epoch_header(e, E, verbose=False, mode='train'):
    print(f"┌{'─'*77}┐")
    if mode == 'train':
        print(f"│{f'STARTING TRAIN EPOCH {e}/{E}':^77s}│")
    elif mode == 'val':
        print(f"│{f'STARTING VALIDATION EPOCH {e}/{E}':^77s}│")
    elif mode == 'test':
        print(f"│{f'STARTING TEST EPOCH {e}/{E}':^77s}│")
    if verbose:
        print(f"├{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*12}┤")
        print(f"│{'Epoch':^12s}│{'Batch':^12s}│{'Target':^12s}│{'Prediction':^12s}│{'Correct?':^12s}│{'Loss':^12s}│")
        print(f"├{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*12}┤")
    else:
        print(f"├{'─'*77}┤")

def pprint_verbose_table(e,E,b,B,label,pred,loss):
    for i in range(len(label)):
        print(f"│{f'{e}/{E}':^12s}│{f'{b}/{B}':^12s}│{label[i]:^12d}│{pred[i]:^12d}│{str(bool(label[i]==pred[i])):^12s}│{loss:^12.6f}│")

def pprint_epoch_footer(e, E, acc, loss, verbose=False, mode='train'):
    if verbose:
        print(f"├{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┤")
    if mode == 'train':
        print(f"│{f'COMPLETED TRAIN EPOCH {e}/{E}':^77s}│")
    elif mode == 'val':
        print(f"│{f'COMPLETED VALIDATION EPOCH {e}/{E}':^77s}│")
    elif mode == 'test':
        print(f"│{f'COMPLETED TEST EPOCH {e}/{E}':^77s}│")
    print(f"├{'─'*77}┤")
    print(f"│{f'Accuracy = {acc}    Loss = {loss}':^77s}│")
    print(f"└{'─'*77}┘")

def pprint_results(test_acc, total_param_size=None, trainable_param_size=None, testing_time=None, EER_random=None, EER_skilled=None):
    print(f"┌{'─'*77}┐")
    print(f"│{'RESULTS':^77s}│")
    print(f"├{'─'*50}┬{'─'*26}┤")
    print(f"│{f'Test Accuracy':^50s}│{f'{test_acc}':^26s}│")
    if total_param_size is not None:
        print(f"├{'─'*50}┼{'─'*26}┤")
        print(f"│{f'Total Parameters':^50s}│{f'{total_param_size}':^26s}│")
    if trainable_param_size is not None:
        print(f"├{'─'*50}┼{'─'*26}┤")
        print(f"│{f'Trainable Parameters':^50s}│{f'{trainable_param_size}':^26s}│")
    if testing_time is not None:
        print(f"├{'─'*50}┼{'─'*26}┤")
        print(f"│{f'Testing Time (microsecs)':^50s}│{f'{testing_time}':^26s}│")
    if EER_random is not None:
        print(f"├{'─'*50}┼{'─'*26}┤")
        print(f"│{f'EER on Random Forgery':^50s}│{f'{EER_random}':^26s}│")
    if EER_skilled is not None:
        print(f"├{'─'*50}┼{'─'*26}┤")
        print(f"│{f'EER on Skilled Forgery':^50s}│{f'{EER_skilled}':^26s}│")
    print(f"└{'─'*50}┴{'─'*26}┘")

def pprint_verification_start(mode='random'):
    print(f"┌{'─'*77}┐")
    if mode == 'random':
        print(f"│{f'STARTING VERIFICATION ON RANDOM FORGERY':^77s}│")
    elif mode == 'skilled':
        print(f"│{f'STARTING VERIFICATION ON SKILLED FORGERY':^77s}│")
    print(f"├{'─'*77}┤")

def pprint_verification_end(eer):
    print(f"│{f'Equal Error Rate = {eer}':^77s}│")
    print(f"└{'─'*77}┘")

def pprint_line():
    print(f"")
