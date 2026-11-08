function methods = getPlottingConstants()
% Define unique colors as variables with informative names
red_color = [1 0 0];
blue_color = [0 0 1];
black_color = [0 0 0];
orange_color = [0.9290 0.6940 0.1250];  
green_color = [0.4660 0.6740 0.1880];   
purple_color = [0.4940 0.1840 0.5560];  
cyan_color = [0.3010 0.7450 0.9330];    
yellow_color = [0.9290 0.8940 0.1250]; 
maroon_color  = [0.6350 0.0780 0.1840];
navy_color    = [0 0.2470 0.5410];
teal_color    = [0 0.5 0.5];
olive_color   = [0.4660 0.5740 0.1880];
pink_color    = [1.00, 0.08, 0.58];
brown_color = [0.45, 0.25, 0.08];

% Define the list of methods as struct array
methods(1).name = 'A-design';  
methods(1).marker = '>';
methods(1).linestyle = '--';
methods(1).color = red_color;  

methods(2).name = 'E-design';  
methods(2).marker = '+';
methods(2).linestyle = ':';
methods(2).color = blue_color; 

methods(3).name = 'D-design'; 
methods(3).marker = '*';
methods(3).linestyle = '-.';
methods(3).color = orange_color; 

methods(4).name = 'T-design';  
methods(4).marker = '<';
methods(4).linestyle = ':';
methods(4).color = cyan_color;  

methods(5).name = 'LR-design'; 
methods(5).marker = 'square';
methods(5).linestyle = '-.';
methods(5).color = orange_color;

methods(6).name = 'Random';  
methods(6).marker = '^';
methods(6).linestyle = '-.';
methods(6).color = black_color; 

methods(7).name = 'b. CRB';  
methods(7).marker = '*';
methods(7).linestyle = '--';
methods(7).color = olive_color;  

methods(8).name = 'WC-MSE';  
methods(8).marker = '<';
methods(8).linestyle = '--';
methods(8).color = teal_color; 

methods(9).name = 'BMSE'; 
methods(9).marker = 'x';
methods(9).linestyle = '--';
methods(9).color = cyan_color; 

methods(10).name = 'WC-BMSE';  
methods(10).marker = 'diamond';
methods(10).linestyle = '--';
methods(10).color = purple_color;
end