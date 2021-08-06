path_to_existing_model = "./Vensim models/model_thesis_V51_influence_np.mdl"  #
# zonder lookup proberen
path_to_new_model = "./Vensim models/model_thesis_V51_correct.mdl"


#line = open(fileSpecifyingError).read()

# line = b'Annual consumption predator:1.4105227152504443, Annual fish ' \
#        b'consumption per capita:1.5306499734836017e-05, Average sinking time:376.5098609798203, Average weight per adult MF:9.946982641907876e-07, Average weight per juvinile MF:5.957082060263719e-08, Average weight per juvinile predator:0.08404829226997838, Average weight per predator:0.2736507339679565, C content copepods:0.4341381500838948, Carbon loss at living depth:0.2972740518532279, Carbon loss underway:0.043834700747921625, Catchability mauriculus:0.1688549582742528, Catchability myctophidae:0.13716685149490365, Consumption by MF in bodyweight:8.342277181413369, Consumption by zooplankton in bodyweight:3.3088404887602834, Costs regular fish:403389122940.81995, Costs status quo mauriculus:192013670006.97983, Costs status quo myctophidae:534913968987.62836, Delay sedimentation:9694.755564758978, Delay weathering:8522.097548850847, Depth euphotic zone:89.56643252950974, Downwelling water:1441637988509187.5, Export efficiency:0.8946889000649549, Female fraction:0.5019565724740257, Fishmeal to fish factor:3.609767130443473, Fraction grazed C ending up in surface:0.24398135421760145, Fraction of migrating MF:0.4051409414717402, Fraction spawning mauriculus vs myctophidae:0.6597240296812314, Grazing in surface by MF:0.4728788347111792, Growth period mauriculus:0.9381964990462569, Growth period myctophidae:0.5091340031115238, Growth period predator:3.5263739719182463, Harvest information delay:2.4733527128624915, Information delay risk perception:5.198850151890361, Initial juvinile predator weight:6.916192255569111, Initial phytoplankton:0.8326268794930562, Initial predator weight:3.5830810266871853, Initial sediment C:3533.8163654122623, Initial surface C:689.0242240405603, Initial weight mauriculus adult:0.29034420376142345, Initial weight mauriculus juvinile:0.06678361108270767, Initial weight myctophidae adult:10.600932520415814, Initial weight myctophidae juvinile:1.3402104058308946, Initial zooplankton:3.444301071716721, Life expectancy mauriculus:2.108317295859491, Life expectancy myctophidae adult:4.051060539988614, Living depth mauriculus:250, Living depth myctophidae:550, Other food sources:2.2519413064943485, Percentage discarded fish:0.0922111484238754, Predator life expectancy:6.819925561051167, Proposed harvesting quota:9.572163575835805, Residence time deep carbon:838.8099230590695, Sale price regular fish:3231923250927.542, Share of aquaculture:0.5163423316758053, Share of irreplaceable fishmeal:0.11151087014302809, Spawning fraction:0.15317586086446167, Specialist capacity building time:5.625252205979434, Surface ocean:357241810881286.9, Survived larvea:146.9766544694778, Switch influence CO2 on phytoplankton:1, Switch influence sunlight on phytoplankton:1, Switch population growth:3, Switch price change:2, Switch risk perception biomass:1, Switch risk perception climate:3, Switch technological innovation MF fisheries:2, Total atmospheric volume:3.721408386964065e+18, Transfer velocity for GtC per year:1.1296731241775415, Turnover time phytoplankton:0.06267617944856882, Upwelling delay surface:7.637464408659595, Vertical migration besides mauriculus:3.0031293191696053, ppm conversion for ocean:2.3753898476095796, policy:None'
line= b'Annual consumption predator:1.3907536057117573, Annual fish consumption per capita:1.469474737620205e-05, Average sinking time:323.2200796123737, Average weight per adult MF:8.492870374889294e-07, Average weight per juvinile MF:6.526754639847289e-08, Average weight per juvinile predator:0.08402188357528234, Average weight per predator:0.2635780832245908, C content copepods:0.48367443179099, Carbon loss at living depth:0.3653449186962435, Carbon loss underway:0.039254776989121734, Catchability mauriculus:0.24303514510228524, Catchability myctophidae:0.17324722742583998, Consumption by MF in bodyweight:8.353426273063254, Consumption by zooplankton in bodyweight:2.4729294899860568, Conversion factor to ppm:2.080564937760703, Costs regular fish:300135441095.9177, Costs status quo mauriculus:263611384478.724, Costs status quo myctophidae:461900275108.45154, Delay sedimentation:10495.408326149349, Delay weathering:10763.016035681721, Depth euphotic zone:108.89689404134015, Downwelling water:1555318353658673.0, Efficiency factor fisheries:7.91498666780805, Export efficiency:0.6888885039259507, Female fraction:0.52460453724168, Fishmeal to fish factor:3.467400588641983, Fraction grazed C ending up in surface:0.27159248293168664, Fraction of migrating MF constant:0.46040213758306286, Fraction spawning mauriculus vs myctophidae:0.8237052969027807, Grazing in surface by MF:0.5003057707984306, Growth period mauriculus:1.0706136625455633, Growth period myctophidae:0.5326993765531479, Growth period predator:3.130859321458165, Harvest information delay:2.9278050558481397, Information delay risk perception:4.687075079640937, Initial juvinile predator weight:6.52621480532144, Initial phytoplankton:1.055838236432402, Initial predator weight:3.489303936509473, Initial sediment C:3027.2008769571144, Initial surface C:573.6535125357823, Initial weight adult MF:9.031198346159341, Initial weight juvinile MF:1.260639583542987, Initial zooplankton:3.300565737715948, Life expectancy mauriculus:2.0219176667653307, Life expectancy myctophidae adult:4.034456938192242, Living depth mauriculus:450, Living depth myctophidae:650, Other carbon fluxes:5.235318898494883, Other food sources:1.9359346291158572, Percentage discarded fish:0.07096117262499896, Predator life expectancy:6.02746987165028, Residence time deep carbon:1000.3329975363536, SWITCH lanternfish to mauriculus:0.43080047447114866, Sale price regular fish:3383501966438.2886, Share of aquaculture:0.4563513881445654, Share of irreplaceable fishmeal:0.965808364963084, Spawning fraction:0.2152916472352292, Specialist capacity building time:5.9318775055936195, Surface ocean:358097907040031.44, Survived larvea:162.03973597795868, Switch influence CO2 on phytoplankton:1, Switch influence sunlight on phytoplankton:1, Switch population growth:2, Switch price change:1, Switch profitability change MF fisheries:2, Switch risk perception biomass:1, Switch risk perception climate:2, Switch risk reward mechanism:2, Total atmospheric volume:3.359846937625056e+18, Transfer velocity for GtC per year:1.1034221818020953, Turnover time phytoplankton:0.08356247815642193, Upwelling delay surface:8.304735969543744, ppm conversion for ocean:1.9351666935407195, policy:None'

# line= b'Annual consumption predator:1.7583777237858835, Annual fish ' \
#       b'consumption per capita:1.4278615045612396e-05, Average sinking time:434.25064567086554, Average weight per adult MF:1.1528662520133687e-06, Average weight per juvinile MF:6.932370064969344e-08, Average weight per juvinile predator:0.07270649884892695, Average weight per predator:0.3024932767723099, C content copepods:0.4894162303019786, Carbon loss at living depth:0.27812163587179817, Carbon loss underway:0.049101139352351254, Catchability mauriculus:0.2781522861477758, Catchability myctophidae:0.162990281562384, Consumption by MF in bodyweight:5.952018538371241, Consumption by zooplankton in bodyweight:3.146039123680466, Costs regular fish:377732748652.26276, Costs status quo mauriculus:149468005095.81717, Costs status quo myctophidae:418441073566.03204, Delay sedimentation:10757.911254015977, Delay weathering:11018.880770846625, Depth euphotic zone:77.15723518273336, Downwelling water:1495613980106179.5, Export efficiency:0.6977514396074911, Female fraction:0.5665322300666303, Fishmeal to fish factor:4.402359531764327, Fraction grazed C ending up in surface:0.3582073334456385, Fraction of migrating MF:0.3786442848410493, Fraction spawning mauriculus vs myctophidae:0.8637726410179383, Grazing in surface by MF:0.4430561761526908, Growth period mauriculus:1.08104444553145, Growth period myctophidae:0.5031078248462921, Growth period predator:3.111885092056172, Harvest information delay:2.7232135718337127, Information delay risk perception:4.265011995107765, Initial juvinile predator weight:5.755718223145077, Initial phytoplankton:1.0311647609201617, Initial predator weight:3.848675746816227, Initial sediment C:3697.1036238598417, Initial surface C:621.8393267007374, Initial weight mauriculus adult:0.39138520220015827, Initial weight mauriculus juvinile:0.05855133511198227, Initial weight myctophidae adult:11.594432998832108, Initial weight myctophidae juvinile:0.9532461776457628, Initial zooplankton:3.8732886267341975, Life expectancy mauriculus:2.29159718902131, Life expectancy myctophidae adult:4.638495468884943, Living depth mauriculus:450, Living depth myctophidae:850, Other food sources:1.9546491843374563, Percentage discarded fish:0.06986759273130798, Predator life expectancy:6.504156551665158, Residence time deep carbon:1075.5963548538725, Sale price regular fish:3442965393309.405, Share of aquaculture:0.44002651800410664, Share of irreplaceable fishmeal:0.6983798770727914, Spawning fraction:0.1737635264807499, Specialist capacity building time:5.512229825864087, Surface ocean:342697235481557.3, Survived larvea:146.98991627551078, Switch influence CO2 on phytoplankton:1, Switch influence sunlight on phytoplankton:1, Switch population growth:3, Switch price change:1, Switch risk perception biomass:1, Switch risk perception climate:3, Switch technological innovation MF fisheries:1, Total atmospheric volume:4.243793492015516e+18, Transfer velocity for GtC per year:1.0907271676190102, Turnover time phytoplankton:0.08124885833752339, Upwelling delay surface:8.540928070138785, Vertical migration besides mauriculus:2.4466814904397602, ppm conversion for ocean:2.146187270219374, policy:None'


# we assume the case specification was copied from the logger
splitOne = line.split(b',') # hier gaat het geloof ik al fout
variables = {}

# -1 because policy entry needs to be removed
for entry in splitOne[0:-1]:
    variable, value = entry.split(b':')
    # Delete the spaces and other rubish on the sides of the variable name
    variable = variable.strip()
    variable = variable.lstrip(b"'")
    variable = variable.rstrip(b"'")

    valueElement = value.strip()
    variables[variable] = value
print(variables)

# This generates a new (text-formatted) model
changeNextLine = False
setted_values = set()

with open(path_to_new_model, 'wb') as new_model:
    skip_next_line = False

    for line in open(path_to_existing_model, 'rb'):
        if skip_next_line:
            skip_next_line = False
            line = b'\n'

        if line.find(b"=") != -1:
            variable = line.split(b"=")[0]
            variable = variable.strip()

            try:
                value = variables[variable]
            except KeyError:
                pass
            else:
                line = variable + b" = " + value
                setted_values.add(variable)
                skip_next_line = True

        new_model.write(line)

notSet = set(variables.keys()) - setted_values
print(notSet)