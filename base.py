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
    

if __name__=="__main__":
    main()