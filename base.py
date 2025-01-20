import workers.worker as worker
import analytics.data_analysis as data_analysis
import analytics.data_analysis_1 as Data_Analysis_1
import analytics.breakpoint as Breakpoint

def main():
    worker.CDL_Worker().init()
    # data_analysis.Data_Analysis().init()
    # analysis = Data_Analysis_1.Data_Analysis_1()
    # analysis = Breakpoint.Breakpoint()
    # analysis.init()

if __name__=="__main__":
    main()
    
    
    
    # 'ctl_game_count', 'hp_game_count','hp_game_count', 'snd_game_count', 
    # 'ctl_bp_rating', 'hp_bp_rating', 'non_traded_kills', 'snd_bp_rating', 'snd_kd',         'snd_kpr',
    
    
    # 'one_v_four_win_count', 'one_v_one_win_count',
    # 'one_v_three_win_count', 'one_v_two_win_count', 'plant_count',
    # 'snd_bp_rating', 'snd_rounds', 'snipe_count', 'zone_capture_count',
    # 'zone_tier_capture_count', 'ctl_attack_rounds', 'hp_kills', 'snd_kills',
    # 'ctl_kills', 'hp_damage', 'hp_deaths', 'snd_damage', 'snd_deaths',
    # 'ctl_damage', 'ctl_deaths', 'hp_gametime', 'ctl_gametime', 'bp_rating',
    # 'slayer_rating', 'dmg_per_min', 'first_blood_percentage', 'kd',
    # 'hp_dmg_10m', 'hp_kd', 'hp_k_10m', 'hp_obj_10m', 'k_p_10m',
    # 'snd_damage_per_round', 'snd_kd', 'snd_kpr',
    # 'snd_plants_defuses_per_round', 'snd_plants_defuses', 'ctl_kd',
    # 'ctl_k_10m', 'ctl_dmg_10m', 'ctl_obj_10m', 'hp_eng_10m', 'ctl_eng_10m',
    # 'tes', 'ctl_t_atrd'