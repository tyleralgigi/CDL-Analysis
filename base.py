import workers.worker as worker
import analytics.data_analysis as data_analysis
import analytics.data_analysis_1 as Data_Analysis_1
import analytics.breakpoint as Breakpoint

def main():
    # worker.CDL_Worker().init()
    # data_analysis.Data_Analysis().init()
    # analysis = Data_Analysis_1.Data_Analysis_1()
    analysis = Breakpoint.Breakpoint()
    analysis.init()
    
main()
    


#####

# TODO
# 1. get team stats from individual breakingpoint pages including pick and vetos and map records
# 2. add to the algothrim to predict what maps will be played betweenn teams
# 3. Change algothirm to predict per map outcomes, so based on what 5 maps are played for the series who will win what maps and series