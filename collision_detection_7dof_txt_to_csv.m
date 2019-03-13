RawData = load('training_data_episode_222_.txt');

num_input = 28;
num_time_step = 5;
for h=1:num_time_step-1
   RawData(h,:) = 0.0; 
end
for k=num_time_step:size(RawData,1)
    RawData(k,num_input*num_time_step+2) = RawData(k,85);
    RawData(k,num_input*num_time_step+3) = RawData(k,86);
    for i=2:num_time_step
        for j=1:num_input
            RawData(k,num_input*(i-1)+j+1) = RawData(k-i+1,j+1);
        end
    end
end
Training = RawData(randperm(fix(size(RawData,1)*1.0)),:);
Validation = RawData(randperm(fix(size(RawData,1)*1.0)),:);
Testing = RawData(randperm(fix(size(RawData,1)*1.0)),:);

csvwrite('raw_data_.csv', RawData);
%csvwrite('training_data_.csv', Training);
%csvwrite('validation_data_.csv', Validation);
%csvwrite('testing_data_.csv', Testing);
