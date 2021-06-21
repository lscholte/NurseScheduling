#Schedule A: Schedule receiving element
#Schedule B: Schedule sending element
#Index: index of element in schedule array
#Dim: length of schedule array
def makeSwitch(scheduleA, scheduleB, index, dim)
    j = index

    shift = index % 28
    shiftArrayBlack = []
    shiftArrayWhite = []
    shiftIndex = []
    for q in range(0, dim):
        if q % 28 == shift:
            shiftIndex.append(q)
            shiftArrayBlack.append(scheduleA[q])
            shiftArrayWhite.append(scheduleB[q])


    nurse = 0
    for k in range(len(shiftIndex)):
        if shiftIndex[k] == index:
            nurse = k
    

    nursesWeek = []
    nursesWeekIndex = []
    for h in range(dim):
        if shift <= 13:
            if 28 * nurse <= h and h < 28 * nurse + 14:
                nursesWeek.append(scheduleA[h])
                nursesWeekIndex.append(h)
        elif shift > 13: 
            if 28 * nurse + 14 <= h and h < 28 * (nurse + 1):
                nursesWeek.append(scheduleA[h])
                nursesWeekIndex.append(h)

    shiftNum = 0
    for i in range(len(nursesWeek)):
        shiftNum += nursesWeek[i]
    
    dayShifts = 0
    for i in range(len(nursesWeek)):
        if i % 2 == 0:
            dayShifts += nursesWeek[i]

    # Better universe nurse is working, worse isn't
    # Check if the nurse you are removing will still have >= 1 day shift that week
    # Check if the nurse you are giving work will have <= 4 shifts that week
    if shiftArrayBlack[nurse] < shiftArrayWhite[nurse]:
        validToReplaceIndex = []
        validToReplaceWithDay = []
        for i in range(len(shiftIndex)):
            if i != nurse and shiftArrayBlack[i] == shiftArrayWhite[nurse]:
                # If the shift is a night shift, any nurse can be moved off the shift 
                if shift % 2 != 0:
                    validToReplaceIndex.append(shiftIndex[i])
                # If the shift is a day shift, check that a nurse will still have >= 1 day shift that week
                # before moving off
                else:
                    weekShifts = []
                    for h in range(dim):
                        if 28 * i <= h and h < 28 * (i + 1):
                            weekShifts.append(scheduleA[h])
                    # Count number of day shifts nurse has that week 
                    dayshifts = 0
                    for m in range(len(weekShifts)):
                        if m % 2 == 0:
                            dayshifts += weekShifts[m]
                    # If they have more than one day shift, you can move them off it
                    if dayshifts > 1:
                        validToReplaceIndex.append(shiftIndex[i])
                    if dayshifts == 1:
                        validToReplaceIndex.append(shiftIndex[i])
                        validToReplaceWithDay.append(shiftIndex[i])

        random.shuffle(validToReplaceIndex)
        replaced = validToReplaceIndex[0]
        needDay = False
        for i in range(len(validToReplaceWithDay)):
            if validToReplaceWithDay[i] == replaced:
                needDay = True
        replacedNurse = 0
        for i in range(len(shiftIndex)):
            if shiftIndex[i] == replaced:
                replacedNurse = i


        

        # Check if adding this shift for a nurse brings weeks total > 4
        # If it does, give the nurse who was taken off this shift a shift from this nurse 
        if shiftNum == 4 or needDay == True:
            replacedWeek = []
            for h in range(dim):
                if shift <= 13:
                    if 28 * replacedNurse <= h and h < 28 * replacedNurse + 14:
                        replacedWeek.append(scheduleA[h])
                elif shift > 13: 
                    if 28 * replacedNurse + 14 <= h and h < 28 * (replacedNurse + 1):
                        replacedWeek.append(scheduleA[h])                            
            validShiftToSwitch = []
            for i in range(len(nursesWeek)):
                if nursesWeek[i] == shiftArrayWhite[nurse] and replacedWeek[i] != shiftArrayWhite[nurse] and i != shift:
                    if needDay == False:
                        validShiftToSwitch.append(i)
                    else:
                        if i % 2 == 0:
                            validShiftToSwitch.append(i)
            random.shuffle(validShiftToSwitch)
            if len(validShiftToSwitch) > 0:
                scheduleA[replaced] = shiftArrayBlack[nurse]
                scheduleA[j] = shiftArrayWhite[nurse]
                if shift <= 13:
                    indexA = 28 * nurse + validShiftToSwitch[0]
                    indexB = 28 * replacedNurse + validShiftToSwitch[0]
                elif shift > 13:
                    indexA = 28 * nurse + validShiftToSwitch[0] + 14
                    indexB = 28 * replacedNurse + validShiftToSwitch[0] + 14
                scheduleA[indexA] = shiftArrayBlack[nurse]
                scheduleA[indexB] = shiftArrayWhite[nurse]

    # Better universe nurse isn't working, worse is 
    # Check if the nurse you are removing will still have >= 1 day shift that week
    # Check if the nurse you are giving work will still have <= 4 shifts that week
    if shiftArrayWhite[nurse] < shiftArrayBlack[nurse]:
        # Least work - if any other nurses not on that shift already have < 4 shifts that week, 
        # just remove the chosen nurse and give one of those nurses work
        lessThanFourIndex = []
        anyLessThanFour = False
        for i in range(len(shiftIndex)):
            if i != nurse and shiftArrayBlack[i] == shiftArrayWhite[nurse]:
                weekShifts = []
                for h in range(dim):
                    if shift <= 13:
                        if 28 * i <= h and h < 28 * i + 14:
                            weekShifts.append(scheduleA[h])
                    elif shift > 13:
                        if 28 * i + 14 <= h and h < 28 * (i + 1):
                            weekShifts.append(scheduleA[h])
                # Count number of shifts nurse has that week 
                shifts = 0
                for m in range(len(weekShifts)):
                    shifts += weekShifts[m]
                if shifts < 4:
                    anyLessThanFour = True
                    lessThanFourIndex.append(shiftIndex[i])
        if anyLessThanFour:
            random.shuffle(lessThanFourIndex)
            given = lessThanFourIndex[0]
            scheduleA[j] = shiftArrayWhite[nurse]
            scheduleA[given] = shiftArrayBlack[nurse]
        else:
            validToSwitchIndex = []
            for i in range(len(shiftIndex)):
                if i != nurse and shiftArrayBlack[i] == shiftArrayWhite[nurse]:
                    validToSwitchIndex.append(shiftIndex[i])
            random.shuffle(validToSwitchIndex)
            switched = validToSwitchIndex[0]
            switchedNurse = 0
            for h in range (len(shiftIndex)):
                if shiftIndex[h] == switched:
                    switchedNurse = h
            switchedSchedule = []
            switchedIndex = []
            for h in range(dim):
                if shift <= 13:
                    if 28 * switchedNurse <= h and h < 28 * switchedNurse + 14:
                        switchedSchedule.append(scheduleA[h])
                        switchedIndex.append(h)
                elif shift > 13:
                    if 28 * switchedNurse + 14 <= h and h < 28 * (switchedNurse + 1):
                        switchedSchedule.append(scheduleA[h])
                        switchedIndex.append(h)

            switchedScheduleDayShifts = 0
            for b in range(len(switchedSchedule)):
                switchedScheduleDayShifts += switchedSchedule[b]

            #scenarios
            # A day shift is being taken from Nurse A and Nurse A has > 1 day shifts
                # give Nurse A a night shift
            # A day shift is being taken from Nurse A and Nurse A has only 1 day shift
                # give nurse A day shift and give other nurse nurse A's day shift
            # A night shift is being taken from Nurse A and Nurse B has > 1 day shifts
                # give Nurse A any shift
            # A night shift is being taken from Nurse A and Nurse B has only 1 day shift
                # give Nurse A a night shift

            validToSwitchIndexB = []
            for n in range(len(switchedSchedule)):
                if dayShifts > 1 and shift % 2 == 0:
                    if switchedSchedule[n] == shiftArrayBlack[nurse] and n % 2 == 1:
                        if nursesWeek[n] == shiftArrayWhite[nurse]:
                            validToSwitchIndexB.append(switchedIndex[n])
                elif dayShifts == 1 and shift % 2 == 0:
                    if switchedSchedule[n] == shiftArrayBlack[nurse] and n % 2 == 0:
                        if nursesWeek[n] == shiftArrayWhite[nurse]:
                            validToSwitchIndexB.append(switchedIndex[n])
                elif shift % 2 == 1 and switchedScheduleDayShifts > 1:
                    if switchedSchedule[n] == shiftArrayBlack[nurse] and nursesWeek[n] == shiftArrayWhite[nurse]:
                        validToSwitchIndexB.append(switchedIndex[n])
                elif shift % 2 == 1 and switchedScheduleDayShifts == 1:
                    if switchedSchedule[n] == shiftArrayBlack[nurse] and nursesWeek[n] == shiftArrayWhite[nurse]:
                        if n % 2 == 1:
                            validToSwitchIndexB.append(switchedIndex[n])
            if len(validToSwitchIndexB) > 0:
                a = random.randint(0, len(validToSwitchIndexB) - 1)
                ind = validToSwitchIndexB[a]
                indB = 0
                for n in range(len(switchedIndex)):
                    if switchedIndex[n] == ind:
                        indB = nursesWeekIndex[n]
                scheduleA[j] = shiftArrayWhite[nurse]
                scheduleA[indB] = shiftArrayBlack[nurse]
                scheduleA[switched] = shiftArrayBlack[nurse]
                scheduleA[ind] = shiftArrayWhite[nurse]