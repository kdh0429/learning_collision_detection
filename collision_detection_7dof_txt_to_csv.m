RawData = load('training_data_20190226_collision.txt');
for i=2:size(RawData,1)
    RawData(i,23) = (RawData(i,16)-RawData(i-1,16))*1000;
    RawData(i,24) = (RawData(i,17)-RawData(i-1,17))*1000;
    RawData(i,25) = (RawData(i,18)-RawData(i-1,18))*1000;
    RawData(i,26) = (RawData(i,19)-RawData(i-1,19))*1000;
    RawData(i,27) = (RawData(i,20)-RawData(i-1,20))*1000;
    RawData(i,28) = (RawData(i,21)-RawData(i-1,21))*1000;
    RawData(i,29) = (RawData(i,22)-RawData(i-1,22))*1000;
end
Training = RawData(randperm(fix(size(RawData,1)*1.0)),:);
Validation = RawData(randperm(fix(size(RawData,1)*1.0)),:);
Testing = RawData(randperm(fix(size(RawData,1)*1.0)),:);


csvwrite('raw_data_.csv', RawData);
csvwrite('training_data_.csv', Training);
%csvwrite('validation_data_.csv', Validation);
%csvwrite('testing_data_.csv', Testing);