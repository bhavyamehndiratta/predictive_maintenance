from loader import load_train, load_test, summarise

train = load_train('FD001')
test, rul = load_test('FD001')

summarise(train, 'FD001 train')
summarise(test, 'FD001 test')

print(f'\nGround truth RUL shape: {rul.shape}')
print(f'RUL sample (first 5 engines): {rul[:5]}')
print(f'\nTrain sample:')
print(train.head(3).to_string())