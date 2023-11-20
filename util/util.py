def data_loader_size(data_loader):
  for data in data_loader:
    print(len(data_loader))
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)
    break