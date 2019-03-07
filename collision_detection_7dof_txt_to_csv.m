RawData = load('training_data_episode_261.txt');
for i=2:size(RawData,1)
    RawData(i,23) = RawData(i-1,9);
    RawData(i,24) = RawData(i-1,10);
    RawData(i,25) = RawData(i-1,11);
    RawData(i,26) = RawData(i-1,12);
    RawData(i,27) = RawData(i-1,13);
    RawData(i,28) = RawData(i-1,14);
    RawData(i,29) = RawData(i-1,15);
    
    RawData(i,30) = RawData(i-1,16);
    RawData(i,31) = RawData(i-1,17);
    RawData(i,32) = RawData(i-1,18);
    RawData(i,33) = RawData(i-1,19);
    RawData(i,34) = RawData(i-1,20);
    RawData(i,35) = RawData(i-1,21);
    RawData(i,36) = RawData(i-1,22);
end

RawData(1,30) = 0;
RawData(1,31) = 0;
RawData(1,32) = 0;
RawData(1,33) = 0;
RawData(1,34) = 0;
RawData(1,35) = 0;
RawData(1,36) = 0;
    
Training = RawData(randperm(fix(size(RawData,1)*1.0)),:);
Validation = RawData(randperm(fix(size(RawData,1)*1.0)),:);
Testing = RawData(randperm(fix(size(RawData,1)*1.0)),:);


csvwrite('raw_data_.csv', RawData);
%csvwrite('training_data_.csv', Training);
%csvwrite('validation_data_.csv', Validation);
%csvwrite('testing_data_.csv', Testing);