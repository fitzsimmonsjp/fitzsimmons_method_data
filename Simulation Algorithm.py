import csv
import numpy as np
import copy

from smart_open import smart_open
import os
import numpy as np
import datetime as dt
from datetime import datetime
from datetime import date
from datetime import timedelta
from numpy import busday_count
import timestring
from multiprocessing import Pool as ThreadPool
from multiprocessing import Process, Manager
import time
from sklearn.datasets import load_digits
from sklearn.mixture.base import BaseMixture
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture

import os, sys, email,re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from gensim.utils import simple_preprocess, lemmatize
from nltk.corpus import stopwords
from gensim import corpora
from smart_open import smart_open
import pickle
import re
import logging
import ssl
import math


datafile = 'projectfile.xer'
projectname = 'Projectname'

region = 'North West England/North Wales'
niter = 100
niterng = np.arange(1, niter)

epoch = datetime(1899, 12, 30)

def timestamp_microsecond(utc_time):
    td = utc_time - epoch
    assert td.resolution == timedelta(microseconds=1)
    return (td.days)

it = 0
offlist = ['2', '3', '4', '5', '6', '7', '1']
templist = []
templist.append([[b'Project ID'] + [b'Alternative Project ID'] + [b'Project Name (confidential)'] + [b'City (confidential)'] + [b'Region'] + [b'Project Type'] + [b'Project LDA Topic'] + [b'Project Value (at contract)'] + [b'Project Value (at completion)'] + [b'Project Start (Project Details)'] + [b'Poject Start (Earliest Task Date)'] + [b'Status/Data Date'] + [b'Project Finish (Project Details)'] + [b'Project Finish (Latest Task)'] + [b'Task ID (Unique Ref)'] + [b'Task Type'] + [b'Actvity ID (User)'] + [b'Activity Name'] + [b'WBS Unique ID'] + [b'WBS Level'] + [b'WBS Name'] + [b'Tasks at WBS Level'] + [b'Tasks Under Specific WBS'] + [b'Original Duration'] + [b'At Completion Duration'] + [b'% Growth'] + [b'Start Date'] + [b'BL Start Date'] + [b'Actual Start Date'] + [b'Live Start Date'] + [b'BL Finish Date'] + [b'Actual Finish Date'] + [b'Live Finish Date'] + [b'Free Float (hrs)'] + [b'Total Float (hrs)'] + [b'Constraint Type'] + [b'Constraint Date'] + [b'Calendar ID'] + [b'Calendar Name'] + [b'Physical % Complete'] + [b'Project Default Hrs per day'] + [b'Project Duration % Complete'] + [b'Concurrent Activities'] + [b'Percentile'] + [b'Task Category (key word links1)'] + [b'Maximum Temperature (Degrees C)'] + [b'Minimum Temperature (Degrees C)'] + [b'Mean Temperature (Degrees C)'] + [b'Sunshine (Total hours)'] + [b'Rainfall (mm)'] + [b'Raindays>=1.0mm'] + [b'Days of Air Frost'] + [b'Market Conditions at time of start'] + [b'Activity Relationships']])

# Step 1: Mine the data from the programme file

#get project data
f = smart_open(datafile)
tasks = 'false'
header = 'false'
projlist1 = []
for l3 in f:
    header = 'false'
    proc = l3.split(b'\t')
    if proc[0] == b'%T' and proc[1] == b'PROJECT\r\n':
        tasks = 'true'
        header = 'true'
    else:
        if proc[0] == b'%T':
            tasks = 'false'
    if tasks == 'true' and header == 'false':
        projlist1.append(proc)

# find column references
c2  = 0
for row in projlist1[1:]:
    c2 += 1
    c1 = 0
    for c in projlist1[0]:
        if c == b'proj_id':
            projid = projlist1[c2][c1]
            projcol = c1
        if c == b'proj_short_name':
            altprojname = projlist1[c2][c1]
        if c == b'last_recalc_date':
            datadate = projlist1[c2][c1]
        if c == b'export_flag':
            exportflag = projlist1[c2][c1]
            expcol = c1
        if c == b'plan_start_date':
            pstartdate = projlist1[c2][c1]
        if c == b'plan_end_date':
            penddate = projlist1[c2][c1]
        if c == b'clndr_id':
            clndr_id = projlist1[c2][c1]
        c1 += 1
    if exportflag == b'Y':
        f = smart_open(datafile)
        # pull out calendars
        tasks = 'false'
        header = 'false'
        cal_list = []
        for l3 in f:
            header = 'false'
            proc = l3.split(b'\t')
            if proc[0] == b'%T' and proc[1] == b'CALENDAR\r\n':
                tasks = 'true'
                header = 'true'
            else:
                if proc[0] == b'%T':
                    tasks = 'false'
            if tasks == 'true' and header == 'false':
                cal_list.append(proc)
                if clndr_id == b'':
                    hrs_day = str(8)
                else:
                    if proc[1] == clndr_id:
                        hrs_day = str(proc[8].decode('utf-8', errors='ignore'))

        # pull out WBS
        f = smart_open(datafile)
        # pull out calendars
        tasks = 'false'
        header = 'false'
        list4 = []
        for l3 in f:
            header = 'false'
            proc = l3.split(b'\t')
            if proc[0] == b'%T' and proc[1] == b'PROJWBS\r\n':
                tasks = 'true'
                header = 'true'
            else:
                if proc[0] == b'%T':
                    tasks = 'false'
            if tasks == 'true' and header == 'false' and proc[2] == projid:
                list4.append(proc)
            else:
                if tasks == 'true' and header == 'false' and proc[0] == b'%F':
                    list4.append(proc)

        c1 = 0
        for c in list4[0]:
            if c == b'wbs_name':
                wbsnamecol = c1
            c1 += 1

        # unload:
        list3 = []

        f = smart_open(datafile)

        # pull out activities
        tasks = 'false'
        header = 'false'
        c1 = 0
        for line in f:
            header = 'false'
            proc = line.split(b'\t')
            if proc[0] == b'%T' and proc[1] == b'TASK\r\n':
                tasks = 'true'
                header = 'true'
            else:
                if proc[0] == b'%T':
                    tasks = 'false'
            if tasks == 'true' and header == 'false':
                c1 += 1
                list3.append(proc)
                if c1 > 1 and proc[2] == projid:
                    hols = 0
                    hols2 = 0
                    calstring = []
                    for item in cal_list:
                        if item[1] == proc[calid]:
                            calstring.append(item)
                    if proc[ac2] == b'' or proc[ac1] == b'':
                        st1 = datetime.strptime(str(proc[pc1]).replace('b\'', '').replace('\'', '')[:26], '%Y-%m-%d %H:%M')
                        fn1 = datetime.strptime(str(proc[pc2]).replace('b\'', '').replace('\'', '')[:26], '%Y-%m-%d %H:%M')
                        st2 = st1
                        fn2 = fn1
                    else:
                        st2 = datetime.strptime(str(proc[ac1]).replace('b\'', '').replace('\'', '')[:26], '%Y-%m-%d %H:%M')
                        fn2 = datetime.strptime(str(proc[ac2]).replace('b\'', '').replace('\'', '')[:26], '%Y-%m-%d %H:%M')
                        st1 = datetime.strptime(str(proc[pc1]).replace('b\'', '').replace('\'', '')[:26], '%Y-%m-%d %H:%M')
                        fn1 = datetime.strptime(str(proc[pc2]).replace('b\'', '').replace('\'', '')[:26], '%Y-%m-%d %H:%M')
                    dtrng = np.arange(timestamp_microsecond(st1), timestamp_microsecond((fn1)) + 1)
                    dtrng2 = np.arange(timestamp_microsecond(st2), timestamp_microsecond((fn2)) + 1)

                    if calstring == []:
                        hols = 0
                        dstring = '1111100'
                    else:
                        for d in dtrng:
                            for s in calstring[0]:
                                if str(d) + '()' in str(s):
                                    hols += 1
                        for d in dtrng2:
                            for s in calstring[0]:
                                if str(d) + '()' in str(s):
                                    hols2 += 1

                        dstring = ''
                        rt = np.arange(0, 7)
                        for r in rt:
                            if '||' + offlist[r] + '()()' in str(s):
                                dstring += '0'
                            else:
                                dstring += '1'

                    # get planned duration
                    st1_1 = timestring.Date(st1)
                    fn1_1 = timestring.Date(fn1)
                    st1_2 = date(st1_1.year, st1_1.month, st1_1.day)
                    fn1_2 = date(fn1_1.year, fn1_1.month, fn1_1.day)
                    oduration = busday_count(st1_2, fn1_2, weekmask=dstring) - hols

                    # get actual duration
                    st2_1 = timestring.Date(st2)
                    fn2_1 = timestring.Date(fn2)
                    st2_2 = date(st2_1.year, st2_1.month, st2_1.day)
                    fn2_2 = date(fn2_1.year, fn2_1.month, fn2_1.day)
                    aduration = busday_count(st2_2, fn2_2, weekmask=dstring) - hols2

                    # define growth percentage
                    if oduration != 0:
                        dgrowth = (aduration / oduration) - 1
                    else:
                        dgrowth = 0

                    # find WBS name
                    for wbs in list4:
                        if wbs[1] == proc[wbsidcol]:
                            wbsname = wbs[wbsnamecol]

                    # build task list (templist)
                    templist.append([[projid] + [altprojname] + [projectname.encode()] + [''.encode()] + [region.encode()] + [''.encode()] + [''.encode()] + [''.encode()] + [''.encode()] + [pstartdate] + [''.encode()] + [datadate] + [penddate] + [''.encode()] + [proc[Utaskid]] + [proc[ttype]] + [proc[taskid]] + [proc[taskname]] + [proc[wbsidcol]] + [''.encode()] + [wbsname] + [''.encode()] + [''.encode()] + [str(oduration).encode()] + [str(aduration).encode()] + [str(dgrowth).encode()] + [''.encode()] + [''.encode()] + [proc[ac1]] + [proc[pc1]] + [''.encode()] + [proc[ac2]] + [proc[pc2]] + [proc[freefloat]] + [proc[totalfloat]] + [proc[consttype]] + [proc[constdate]] + [proc[calid]] + [''.encode()] + [proc[pperc]] + [hrs_day.encode()] + [''.encode()] + [''.encode()] + [''.encode()] + [''.encode()] + [''.encode()] + [''.encode()] + [''.encode()] + [''.encode()] + [''.encode()] + [''.encode()] + [''.encode()] + [''.encode()] + [''.encode()]])

                    list3[c1 - 1] += [str(dgrowth).encode()]
                else:
                    #find columnns
                    c3 = 0
                    for c in list3[0]:
                        if c == b'wbs_id':
                            wbsidcol = c3
                        if c == b'act_start_date':
                            ac1 = c3
                        if c == b'act_end_date':
                            ac2 = c3
                        if c == b'free_float_hr_cnt':
                            freefloat = c3
                        if c == b'total_float_hr_cnt':
                            totalfloat = c3
                        if c == b'target_start_date':
                            pc1 = c3
                        if c == b'target_end_date':
                            pc2 = c3
                        if c == b'cstr_date':
                            constdate = c3
                        if c == b'cstr_type':
                            consttype = c3
                        if c == b'task_code':
                            taskid = c3
                        if c == b'task_id':
                            Utaskid = c3
                        if c == b'task_name':
                            taskname = c3
                        if c == b'clndr_id':
                            calid = c3
                        if c == b'task_type':
                            ttype = c3
                        if c == b'phys_complete_pct':
                            pperc = c3
                        if c == b'phys_complete_pct':
                            pperc = c3
                        c3 += 1

        task_list = []
        for y in templist:
            line2 =[]
            for y2 in y:
                for x in y2:
                    line2 += [x.decode('utf-8', errors='ignore')]
                task_list.append(line2)

        templist = []
        list1 = []

        f = smart_open(datafile)
        # pull out links
        tasks = 'false'
        header = 'false'
        for l3 in f:
            header = 'false'
            problems = 'false'
            proc = l3.split(b'\t')
            if proc[0] == b'%T' and proc[1] == b'TASKPRED\r\n':
                tasks = 'true'
                header = 'true'
            else:
                if proc[0] == b'%T':
                    tasks = 'false'
                if proc[0] == b'%E\r\n':
                    problems = 'true'
            if problems == 'false':
                if tasks == 'true' and header == 'false' and proc[4] == projid:
                    line3 = []
                    c5 = 0
                    for line2 in proc:
                        if c5 < 8:
                            line3 += [line2]
                        c5 += 1
                    list1.append(line3)

        link_list = []
        for y in list1:
            line2 = []
            for x in y:
                line2 += [x.decode('utf-8')]
            link_list.append(line2)

        # unload:
        list2 = []
        list1 = []

for line in link_list:
    if line[7] == '':
        line[7] = '0'

print('Finished gathering data from the programme file')

# Step 1B: Calculate Earliest Start and Latest Finish
dfs = '%Y-%m-%d %H:%M'
st_1 = '2080-01-01 08:00'
st_1 = datetime.strptime(st_1[:26], '%Y-%m-%d %H:%M')
fn_1 = '1900-01-01 08:00'
fn_1 = datetime.strptime(fn_1[:26], '%Y-%m-%d %H:%M')
for line in task_list[1:]:
    if line[28] == '':
        dt1 = line[29]
    else:
        dt1 = line[28]
    if line[31] == '':
        dt2 = line[32]
    else:
        dt2 = line[31]
    dt1 = datetime.strptime(dt1[:26], '%Y-%m-%d %H:%M')
    dt2 = datetime.strptime(dt2[:26], '%Y-%m-%d %H:%M')

    if dt1 < st_1:
        st_1 = dt1
    if dt2 > fn_1:
        fn_1 = dt2

for line in task_list[1:]:
    line[10] = str(st_1)
    line[13] = str(fn_1)
    if line[9] == '' or line[9] == ' ':
        line[9] = str(st_1)
        dfs = '%Y-%m-%d %H:%M:%S'

# unload some lists
pplist = []
pplist2 = []
pplist3 = []

# Step 2: Append average (expected) weather data to tasks

print('Getting weather data')

w = open('Weather Data Means.txt', encoding='utf-8')

list1 = task_list

list2 = []
for line in w:
    ss = line.split('\t')
    list2.append(ss)

w.close()

c_range = np.arange(45, 52)

c3 = 0
c4 = 0
for wt in c_range:
    c4 += 1
    ff = int(wt)
    measure = list1[0][ff]
    for l2 in list1[1:]:
        region = l2[4]
        if l2[28] == '':
            dt1 = l2[29]
        else:
            dt1 = l2[28]
        dt2 = datetime.strptime(dt1[:26], '%Y-%m-%d %H:%M')
        mstart = datetime.strptime(str(dt2.year) + '-' + str(dt2.month) + '-' + '01 00:00'[:26], '%Y-%m-%d %H:%M')
        mm = int(dt2.month)
        c3 += 1
        for l3 in list2:
            if l3[0] == region and int(l3[2]) == mm:
                l2[ff] = l3[ff - 42].replace('\n', '')
                print(str(c4) + ' of 7 ' + str(c3 / (int(list1.__len__())*7)))
                break
    print('finished ' + str(measure))

task_list = list1
list1 = []

print('Finished gathering weather data')

print('Getting economic data...')

# Step 3: Append economic data to project file:
w = open('Economic Data.txt', encoding='utf-8')

list2 = []
for line in w:
    ss = line.split('\t')
    list2.append(ss)

w.close()

c5 = 0
for line in task_list[1:]:
    p_start = datetime.strptime(line[9][:26], dfs)
    ps = timestamp_microsecond(p_start)

    c1 = 0
    for l2 in list2:
        d1 = timestamp_microsecond(datetime.strptime(list2[c1][0][:26], '%d/%m/%Y'))
        if c1 == 86:
            break
        else:
            d2 = timestamp_microsecond(datetime.strptime(list2[c1 + 1][0][:26], '%d/%m/%Y'))
            if ps > d1 and ps < d2:
                line[52] = str(l2[1]).replace('\n', '')
                print(c5/ int(task_list.__len__()))
                break
        c1 += 1
    c5 += 1

print('Finished collecting economic data')

# Step 4: Apply Topic Modelling to Activities

#fetch stop words
sw = open('Stopwords2.txt', encoding='utf-8')
sw1 = []
sw1 += sw
sw.close()

sw2 = []
for line in sw1:
    rr = str(line).replace('\n', '')
    pp = rr.replace('\n', '')
    sw2.append(pp)

st2 = stopwords.words('english')
stop_words = st2 + sw2

#call trained topic models
gmm = pickle.load(open("gmmmodelfinal.sav", 'rb'))
model = pickle.load(open("w2vmodel-anon.sav", 'rb'))
sklearn_pca = PCA(n_components = 2)
ARR1 = sklearn_pca.fit_transform(model.wv.vectors)
t_range = np.arange(0, 20)

pplist3 = []
pplist4 = []

no_valid = ''
for linea in task_list[1:]:
    doc_out = []
    for wd in simple_preprocess(linea[17]):
        if wd not in stop_words:  # remove stopwords
            lemmatized_word = lemmatize(wd, allowed_tags=re.compile('(NN|JJ|RB)'))
            if lemmatized_word:
                lword = str(lemmatized_word[0].split(b'/')[0].decode('utf-8'))
                if lword in model.wv.vocab:
                    doc_out = doc_out + [lword]
            else:
                if wd in model.wv.vocab:
                    doc_out = doc_out + [wd]
        else:
            continue
    if not doc_out:
        if no_valid is 'true':
            xi_topic = '20'
            no_valid = 'true'
        else:
            xi_topic = '20'
            no_valid = 'true'
    else:
        Y2 = model.wv.__getitem__(doc_out)
        ARR2 = sklearn_pca.transform(Y2)
        ARR3 = gmm.predict_proba(ARR2)

        pplist = [[i] for i in ARR3]
        pplist2 = []
        for t in t_range:
            addup = 0
            c3 = 0
            for pp in pplist:
                addup += 1 + pp[0][t] - 1
                c3 += 1
            pplist2 += [addup / c3]

        c1 = 0
        chk = 0
        for line in pplist2:
            if c1 == 0:
                chk = line
                xi_topic = c1
            else:
                if line > chk:
                    chk = line
                    xi_topic = c1
            c1 += 1
        no_valid = 'false'
    pplist3.append(xi_topic)
    if xi_topic == 20:
        pplist4.append([0.0] + [0.0] + [0.0] + [0.0] + [0.0] + [0.0] + [0.0] + [0.0] + [0.0] + [0.0] + [0.0] + [0.0] + [
            0.0] + [0.0] + [0.0] + [0.0] + [0.0] + [0.0] + [0.0] + [0.0])
    else:
        pplist4.append(pplist2)

c1 = 0
for line in pplist3:
    c1 += 1
    task_list[c1][44] = str(line)
    task_list[c1][6] = pplist4[c1 - 1]

print('Finished getting task topics')

print('Getting activity relationships...')

# Step 5: Get link data by activity


def eucreturn(value, sorted_list, column_ref, return_column):

    id1 = int(value)
    ch1b = int(sorted_list.__len__()) - 1
    max1 = int(sorted_list.__len__()) - 1
    found1 = 'false'
    ch2b = (ch1b / 2) * -1
    chkb = 0
    # check top and bottom of the list
    if id1 == sorted_list[sorted_list.__len__() - 1][column_ref]:  # if the value is found at the end of the list
        found1 = sorted_list[sorted_list.__len__() - 1][return_column]
    if id1 == sorted_list[0][column_ref]:  # if the value is found at the beginning of the list
        found1 = sorted_list[0][return_column]
    # look for the value in the list
    while found1 == 'false':
        ch1b = min(max(int(ch1b + ch2b), 0), max1)
        if sorted_list[ch1b][column_ref] == id1:  # if value is found:
            found1 = sorted_list[ch1b][return_column]
            break
        else:
            if sorted_list[ch1b][column_ref] > id1:  # if the search area is higher than the search value
                if ch2b < 0:
                    ch2b = round(ch2b / 2)
                else:
                    ch2b = round((ch2b / 2) * -1)
            else:  # the search area is lower than the search value
                if ch2b < 0:
                    ch2b = round((ch2b / 2) * -1)
                else:
                    ch2b = round(ch2b / 2)
            # the search has finished prematurely and needs to proceed incrementally
            if ch2b == 0:
                if sorted_list[min(ch1b + 1, max1)][column_ref] <= id1:
                    ch2b = 1
                else:
                    if sorted_list[max(ch1b - 1, 0)][column_ref] >= id1:
                        ch2b = -1
        if chkb == ch1b:  # if value is not found:
            found1 = '#N/A'
            break
        chkb = ch1b

    return found1


# create lookup list of topics
list1 = []
for line in task_list[1:]:
    list1.append([int(line[14])] + [line[44]] + [line[23]] + [line[35]] + [line[36]] + [line[28]] + [line[29]])

list1 = sorted(list1, key=lambda x: (x[0], x[0]), reverse=False)

for line in link_list:
    line[0] = eucreturn(line[3], list1, 0, 1)
    line[1] = eucreturn(line[3], list1, 0, 2)
    line[2] = int(line[2])
    line[3] = int(line[3])
    line[5] = eucreturn(line[2], list1, 0, 2)
    line[7] = line[7].replace('\r\n', '')

si2 = sorted(link_list, key=lambda x: (x[2], x[0]), reverse=False)

counter = 0
for l3 in task_list[1:]:
    counter += 1
    print(counter / (task_list.__len__() - 1))
    first1 = 0
    relstring = ''
    aid1 = int(l3[14])
    ch1 = int(si2.__len__()) - 1
    max2 = int(si2.__len__()) - 1
    found = 'false'
    ch2 = (ch1 / 2) * -1
    chk = 0
    while found == 'false':
        ch1 = min(max(int(ch1 + ch2), 0), max2)
        # if found:
        if si2[ch1][2] == aid1:
            # do what needs to be done
            ch4 = aid1
            ct1 = 0
            # look down
            while ch4 == aid1:
                if (ch1 + ct1) < si2.__len__():
                    ch4 = si2[ch1 + ct1][2]
                    if ch4 == aid1:
                        #do the necessary
                        if first1 == 0:
                            relstring = si2[ch1 + ct1][0] + '|' + str(si2[ch1 + ct1][3]) + '|' + si2[ch1 + ct1][6] + '|' + si2[ch1 + ct1][7].replace('\r\n', '') + '|' + si2[ch1 + ct1][1]
                            first1 = 1
                        else:
                            relstring += ';' + si2[ch1 + ct1][0] + '|' + str(si2[ch1 + ct1][3]) + '|' + si2[ch1 + ct1][6] + '|' + si2[ch1 + ct1][7].replace('\r\n', '') + '|' + si2[ch1 + ct1][1]
                else:
                    break
                ct1 += 1
            # lookup
            ct1 = -1
            ch4 = aid1
            while ch4 == aid1:
                if (ch1 + ct1) >= 0:
                    ch4 = si2[ch1 + ct1][2]
                    if ch4 == aid1:
                        #do the necessary
                        if first1 == 0:
                            relstring = si2[ch1 + ct1][0] + '|' + str(si2[ch1 + ct1][3]) + '|' + si2[ch1 + ct1][6] + '|' + si2[ch1 + ct1][7].replace('\r\n', '') + '|' + si2[ch1 + ct1][1]
                            first1 = 1
                        else:
                            relstring += ';' + si2[ch1 + ct1][0] + '|' + str(si2[ch1 + ct1][3]) + '|' + si2[ch1 + ct1][6] + '|' + si2[ch1 + ct1][7].replace('\r\n', '') + '|' + si2[ch1 + ct1][1]
                else:
                    break
                ct1 -= 1
            found = 'true'
            break
        else:
            if si2[ch1][2] > aid1:
                if ch2 < 0:
                    ch2 = round(ch2 / 2)
                else:
                    ch2 = round((ch2 / 2) * -1)
            else:
                if ch2 < 0:
                    ch2 = round((ch2 / 2) * -1)
                else:
                    ch2 = round(ch2 / 2)
            if ch2 == 0:
                if si2[min(ch1 + 1, max2)][2] <= aid1:
                    ch2 = 1
                else:
                    if si2[max(ch1 - 1, 0)][2] >= aid1:
                        ch2 = -1
        if chk == ch1:
            # print(aid1)
            break
        chk = ch1
    l3[53] = relstring

print('Finished getting activity relationships')

# unload some lists
pplist = []
pplist2 = []
pplist3 = []

# Step 6: Calculate schedule quality score

print('Calculating schedule quality score...')

pfin = datetime.strptime(task_list[1][13][:26], '%Y-%m-%d %H:%M:%S')
pst = datetime.strptime(task_list[1][10][:26], '%Y-%m-%d %H:%M:%S')
dstring = '1111100'
st1_1 = timestring.Date(pst)
fn1_1 = timestring.Date(pfin)
st1_2 = date(st1_1.year, st1_1.month, st1_1.day)
fn1_2 = date(fn1_1.year, fn1_1.month, fn1_1.day)
pduration = busday_count(st1_2, fn1_2, weekmask=dstring)

si2 = sorted(link_list, key=lambda x: (x[2], x[0]), reverse=False)
si3 = sorted(link_list, key=lambda x: (x[3], x[0]), reverse=False)

# go through the list of tasks with no predecessors 1 at a time
# count all predecessors per activity
progress = 0
fail_count = 0
for l3 in task_list[1:]:
    progress += 1
    p_counter = 0
    s_counter = 0
    lag_counter = 0
    lead_counter = 0
    print(progress / task_list.__len__())
    c2 = -1
    aid1 = int(l3[14])
    ch1 = int(si2.__len__()) - 1
    max2 = int(si2.__len__()) - 1
    found = 'false'
    ch2 = (ch1 / 2) * -1
    chk = 0
    sq_pass = 'true'
    while found == 'false':
        ch1 = min(max(int(ch1 + ch2), 0), max2)
        # if found:
        if si2[ch1][2] == aid1:
            # do what needs to be done
            ch3 = ch1
            ch4 = aid1
            ct1 = 0
            checker = 0
            # look down
            while ch4 == aid1:
                if (ch1 + ct1) < si2.__len__():
                    ch4 = si2[ch1 + ct1][2]
                    if ch4 == aid1:
                        p_counter += 1
                        if float(si2[ch1 + ct1][7]) < 0:
                            lead_counter += 1
                        if float(si2[ch1 + ct1][7]) > 0:
                            lag_counter += 1
                else:
                    break
                ct1 += 1
            # lookup
            ct1 = -1
            ch4 = aid1
            checker = 0
            while ch4 == aid1:
                if (ch1 + ct1) >= 0:
                    ch4 = si2[ch1 + ct1][2]
                    if ch4 == aid1:
                        p_counter += 1
                        if float(si2[ch1 + ct1][7]) < 0:
                            lead_counter += 1
                        if float(si2[ch1 + ct1][7]) > 0:
                            lag_counter += 1
                else:
                    break
                ct1 -= 1
            found = 'true'
            break
        else:
            if si2[ch1][2] > aid1:
                if ch2 < 0:
                    ch2 = round(ch2 / 2)
                else:
                    ch2 = round((ch2 / 2) * -1)
            else:
                if ch2 < 0:
                    ch2 = round((ch2 / 2) * -1)
                else:
                    ch2 = round(ch2 / 2)
            if ch2 == 0:
                if si2[min(ch1 + 1, max2)][2] <= aid1:
                    ch2 = 1
                else:
                    if si2[max(ch1 - 1, 0)][2] >= aid1:
                        ch2 = -1
        if chk == ch1:
            break
        chk = ch1

    # count number of successors
    found = 'false'
    while found == 'false':
        ch1 = min(max(int(ch1 + ch2), 0), max2)
        # if found:
        if si3[ch1][3] == aid1:
            # do what needs to be done
            ch3 = ch1
            ch4 = aid1
            ct1 = 0
            checker = 0
            # look down
            while ch4 == aid1:
                if (ch1 + ct1) < si3.__len__():
                    ch4 = si3[ch1 + ct1][3]
                    if ch4 == aid1:
                        s_counter += 1
                else:
                    break
                ct1 += 1
            # lookup
            ct1 = -1
            ch4 = aid1
            checker = 0
            while ch4 == aid1:
                if (ch1 + ct1) >= 0:
                    ch4 = si3[ch1 + ct1][3]
                    if ch4 == aid1:
                        s_counter += 1
                else:
                    break
                ct1 -= 1
            found = 'true'
            break
        else:
            if si3[ch1][3] > aid1:
                if ch2 < 0:
                    ch2 = round(ch2 / 2)
                else:
                    ch2 = round((ch2 / 2) * -1)
            else:
                if ch2 < 0:
                    ch2 = round((ch2 / 2) * -1)
                else:
                    ch2 = round(ch2 / 2)
            if ch2 == 0:
                if si3[min(ch1 + 1, max2)][3] <= aid1:
                    ch2 = 1
                else:
                    if si3[max(ch1 - 1, 0)][3] >= aid1:
                        ch2 = -1
        if chk == ch1:
            break
        chk = ch1

    # perform tests on task
    if p_counter == 0:
        sq_pass = 'false'
    if p_counter > 2:
        sq_pass = 'false'
    if s_counter == 0:
        sq_pass = 'false'
    if lag_counter > 0:
        sq_pass = 'false'
    if lead_counter > 0:
        sq_pass = 'false'
    if l3[34] == '':
        l3[34] = '0'
    else:
        if float(l3[34]) < 0:
            sq_pass = 'false'
    if int(l3[23]) / pduration > 0.1:
        sq_pass = 'false'
    if l3[35] == 'CS_MANDFIN' or l3[35] == 'CS_MANDSTART' or l3[35] == 'CS_MEO' or l3[35] == 'CS_MSO' or l3[
        35] == 'CS_MSOB' or l3[35] == 'CS_MEOB':
        sq_pass = 'false'

    if sq_pass == 'false':
        fail_count += 1

quality_score = str(1 - (fail_count / task_list.__len__()))

for line in task_list[1:]:
    line[43] = quality_score

print('Completed getting schedule quality score - ' + str("%.2f" % float(quality_score)) + '%')

# Step 8: Add up concurrent activities.
print('Gathering concurrent activity information...')

c1 = 0
newlist = []
for line in task_list[1:]:
    c1 += 1
    newlist.append(c1)


def conact(rr):

    if task_list[rr][31] == '':
        st1 = task_list[rr][29]
        fn1 = task_list[rr][32]
    else:
        st1 = task_list[rr][28]
        fn1 = task_list[rr][31]
    st2 = timestamp_microsecond(datetime.strptime((st1)[:26], '%Y-%m-%d %H:%M'))
    fn2 = timestamp_microsecond(datetime.strptime((fn1)[:26], '%Y-%m-%d %H:%M'))
    actid = task_list[rr][14]
    conc_n = 0
    for l2 in task_list[1:]:
        conc = 'false'
        if l2[14] != actid:
            if l2[31] == '':
                st3 = l2[29]
                fn3 = l2[32]
            else:
                st3 = l2[28]
                fn3 = l2[31]
            st4 = timestamp_microsecond(datetime.strptime((st3)[:26], '%Y-%m-%d %H:%M'))
            fn4 = timestamp_microsecond(datetime.strptime((fn3)[:26], '%Y-%m-%d %H:%M'))
            if st4 > st2 and st4 < fn2:
                conc = 'true'
            if st4 < st2 and fn4 > st2:
                conc = 'true'
            if st4 < st2 and fn4 > fn2:
                conc = 'true'
            if conc == 'true':
                conc_n += 1

    conc_n = conc_n / (task_list.__len__() - 1)

    return conc_n


if __name__ == '__main__':
    pool = ThreadPool()
    results = pool.map(conact, newlist)
    pool.close()
    pool.join()

c1 = 0
sumcon = 0
for line in results:
    sumcon += int(line)
    c1 += 1
    task_list[c1][42] = line

avcon = sumcon / c1


print('Finished gathering concurrent activity data')

# Step 9: Simulate Project

savedmodel = "SVM-pc8.sav"
savedscaler = "ScalerSC6.sav"
clf = pickle.load(open(savedmodel, 'rb'))
sc = pickle.load(open(savedscaler, 'rb'))


# Work out average duration
md1 = 0
c1 = 0
for line in task_list[1:]:
    c1 += 1
    md1 += float(line[23])

md1 = md1 / c1

# Work out standard duration deviation
sumsq = 0
c1 = 0
for line in task_list[1:]:
    c1 += 1
    sumsq += (float(line[23]) - md1)**2

stdv = math.sqrt(sumsq / (c1 - 1))

print('stdv = ' + str(stdv) + ' >>> Average = ' + str(md1))

# Gather the conditional probability tables.

w = open('WeatherVar.txt', encoding='utf-8')

weathervar = []
for line in w:
    ss = line.split('\t')
    weathervar.append(ss)

w.close()

w = open('TopicVarTrue.txt', encoding='utf-8')

topvartrue = []
for line in w:
    ss = line.split('\t')
    topvartrue.append(ss)

w.close()

w = open('TopicVarFalse.txt', encoding='utf-8')

topvarfalse = []
for line in w:
    ss = line.split('\t')
    topvarfalse.append(ss)

w.close()

w = open('ConcurrentVar.txt', encoding='utf-8')

concurrentvar = []
for line in w:
    ss = line.split('\t')
    concurrentvar.append(ss)

w.close()

w = open('FloatVar.txt', encoding='utf-8')

floatvar = []
for line in w:
    ss = line.split('\t')
    floatvar.append(ss)

w.close()

w = open('MarketVar.txt', encoding='utf-8')

marketvar = []
for line in w:
    ss = line.split('\t')
    marketvar.append(ss)

w.close()

w = open('AnWeatherAv.txt', encoding='utf-8')

avweather = []
for line in w:
    ss = line.split('\t')
    avweather.append(ss)

w.close()

w = open('priors.txt', encoding='utf-8')

priors = []
for line in w:
    priors.append(line)

w.close()

w = open('SchedQual.txt', encoding='utf-8')

sq_val = []
for line in w:
    sq_val.append(line)

w.close()

tester = ''

# SVM duration calculator
def calc_duration_growth(Xi):

    global tester
    global md1
    global stdv
    aa = []
    for bb in Xi[:3]:
        aa += [float(bb)]
    for bb in Xi[3]:
        aa += [float(bb)]
    Xi2 = np.array(aa)
    Xi2 = Xi2.reshape(1, -1)
    Xi3 = sc.transform(Xi2)
    svmdur = float(clf.predict(Xi3))
    if svmdur < 0:
        tester = "true " + str(svmdur)
    if svmdur < -1:
        svmdur = -0.49
    if svmdur > 0:
        upperlimit = 2 * svmdur
        lowerlimit = 0
    else:
        upperlimit = 0
        lowerlimit = -0.99

    sigma1 = svmdur/3

    randval = np.random.randn()
    if randval >= 0:
        if svmdur < 0:
            overallvar = max(svmdur + (randval * sigma1), -1)
        else:
            overallvar = svmdur + min(randval * sigma1, svmdur)
    else:
        if svmdur < 0:
            overallvar = max(svmdur + max(randval * sigma1, svmdur), -1)
        else:
            overallvar = svmdur + max(randval * sigma1, -1 * svmdur)

    if overallvar > 0.0:
        overallvar = overallvar / (1 + (overallvar * (aa[2] / (md1 + ((1 * stdv) / (stdv / md1))))))
    else:
        overallvar = overallvar / (1 + ((overallvar * (aa[2] / (md1 + ((1 * stdv) / (stdv / md1))))) * -1))

    return overallvar


# Joint probability calculator
def bayestrigger(tasktopic, predtopic, marketb, floatb, concb, topb, w1b, w2b, w3b, w4b, w5b, w6b):

    # Get probabilities for delay
    pw1 = float(weathervar[tasktopic][0 + (12 * w1b)])
    pw2 = float(weathervar[tasktopic][2 + (12 * w2b)])
    pw3 = float(weathervar[tasktopic][4 + (12 * w3b)])
    pw4 = float(weathervar[tasktopic][6 + (12 * w4b)])
    pw5 = float(weathervar[tasktopic][8 + (12 * w5b)])
    pw6 = float(weathervar[tasktopic][10 + (12 * w6b)])
    pt1 = float(topvartrue[tasktopic][predtopic + (21 * topb)])
    pm1 = float(marketvar[tasktopic][0 + (2 * marketb)])
    pf1 = float(floatvar[tasktopic][0 + (2 * floatb)])
    pcc1 = float(concurrentvar[tasktopic][0 + (2 * concb)])

    # Get probabilities for not delay
    pnw1 = float(weathervar[tasktopic][1 + (12 * w1b)])
    pnw2 = float(weathervar[tasktopic][3 + (12 * w2b)])
    pnw3 = float(weathervar[tasktopic][5 + (12 * w3b)])
    pnw4 = float(weathervar[tasktopic][7 + (12 * w4b)])
    pnw5 = float(weathervar[tasktopic][9 + (12 * w5b)])
    pnw6 = float(weathervar[tasktopic][11 + (12 * w6b)])
    pnt1 = float(topvarfalse[tasktopic][predtopic + (21 * topb)])
    pnm1 = float(marketvar[tasktopic][1 + (2 * marketb)])
    pnf1 = float(floatvar[tasktopic][1 + (2 * floatb)])
    pncc1 = float(concurrentvar[tasktopic][1 + (2 * concb)])
    p_d = float(priors[tasktopic])
    p_nd = 1 - p_d

    p_delay2 = (pw1 * pw2 * pw3 * pw4 * pw5 * pw6 * pf1 * pcc1 * pt1 * pm1 * p_d)
    p_ndelay2 = (pnw1 * pnw2 * pnw3 * pnw4 * pnw5 * pnw6 * pnf1 * pncc1 * pnt1 * pnm1 * p_nd)
    p_delay = p_delay2 / (p_delay2 + p_ndelay2)

    r_trigger = np.random.random_sample()

    if r_trigger <= p_delay:
        delay = 'True'
    else:
        delay = 'False'

    return delay


# get the boolean figures for weather
def getweatherbooleans(ew1, ci, regionx):

    global avweather
    for lx in avweather:
        if lx[0] == regionx:
            avx = lx[ci]

    if float(ew1) >= float(avx):
        wbx = 0
    else:
        wbx = 1

    return wbx


# get the boolean figure for concurrent activity
def getconbool(cc1):

    global avcon
    if float(cc1) >= avcon:
        cbx = 0
    else:
        cbx = 1

    return cbx


# get the boolean for market conditions
def getmarketbool(m1x):

    if float(m1x) <= 0:
        mbx = 0
    else:
        mbx = 1

    return mbx


# Define the functions for calculating task variables
def getstartday(contype, condate, t_dur, ltype, pdur1, lag1):

    global dstring
    global epoch
    global st1_2
    cd1 = datetime.strptime(condate[:26], '%Y-%m-%d %H:%M')
    cd_1 = timestring.Date(cd1)
    cd_2 = date(cd_1.year, cd_1.month, cd_1.day)
    cd2 = busday_count(st1_2, cd_2, weekmask=dstring)

    if contype == 'CS_MSOA' or contype == 'CS_MSO' or contype == 'CS_MANDSTART':
        if ltype == 'PR_FS':
            t_start = float(cd2) + float(pdur1) + (float(lag1)/8)
        if ltype == 'PR_FF':
            t_start = float(cd2) + float(pdur1) - float(t_dur) + (float(lag1)/8)
        if ltype == 'PR_SS':
            t_start = float(cd2) + (float(lag1)/8)
        if ltype == 'PR_SF':
            t_start = float(cd2) - float(t_dur) + (float(lag1) / 8)
    else:
        if contype == 'CS_MEOA' or contype == 'CS_MEO' or contype == 'CS_MANDFIN':
            if ltype == 'PR_FS':
                t_start = float(cd2) + (float(lag1)/8)
            if ltype == 'PR_FF':
                t_start = float(cd2) - float(t_dur) + (float(lag1)/8)
            if ltype == 'PR_SS':
                t_start = float(cd2) - float(pdur1) + (float(lag1)/8)
            if ltype == 'PR_SF':
                t_start = float(cd2) - float(pdur1) - float(t_dur) + (float(lag1) / 8)
        else:
            if ltype == 'PR_FS':
                t_start = float(cd2) + float(pdur1) + (float(lag1) / 8)
            if ltype == 'PR_FF':
                t_start = float(cd2) + float(pdur1) - float(t_dur) + (float(lag1) / 8)
            if ltype == 'PR_SS':
                t_start = float(cd2) + (float(lag1) / 8)
            if ltype == 'PR_SF':
                t_start = float(cd2) - float(t_dur) + (float(lag1) / 8)

    return t_start


# define parameters of cumulative early start & cumulative early finish
def getcumes(ltype, lag1, pdur1, edur1):
    cum2 = 0
    if ltype == "PR_FS":
        cum2 = (float(lag1)/8) + float(pdur1)
    else:
        if ltype == "PR_SS":
            cum2 = (float(lag1)/8)
        else:
            if ltype == "PR_FF":
                cum2 = float(pdur1) - float(edur1) + (float(lag1)/8)
            else:
                if ltype == "PR_SF":
                    cum2 = float(pdur1) + (float(lag1) / 8)
    return cum2


# The secondary cumlative ES and EF calculator, where in the middle of a chain
def getcumes2(ltype, lag1, pes, pef, edur1):
    cum2 = 0
    if ltype == "PR_FS":
        cum2 = (float(lag1)/8) + pef
    else:
        if ltype == "PR_SS":
            cum2 = (float(lag1)/8) + pes
        else:
            if ltype == "PR_FF":
                cum2 = pef - float(edur1) + (float(lag1)/8)
            else:
                if ltype == "PR_SF":
                    cum2 = pef + (float(lag1) / 8)
    return cum2


# Get the cumulative late finish position
def getcumlf(ltype, pls, plf, lag, pdur1, edur1):
    cum2 = 0
    if ltype == "PR_FS":
        cum2 = pls - (lag/8)
    else:
        if ltype == "PR_SS":
            cum2 = pls - (lag/8) + float(pdur1)
        else:
            if ltype == "PR_FF":
                cum2 = plf - (lag/8)
            else:
                if ltype == "PR_SF":
                    cum2 = plf + (lag/8) + float(edur1)
    return cum2


# efficient finding binary search algorithm
def eucfind(value, sorted_list, column_ref):

    id1 = int(value)
    ch1b = int(sorted_list.__len__()) - 1
    max1 = int(sorted_list.__len__()) - 1
    found1 = 'false'
    ch2b = (ch1b / 2) * -1
    chkb = 0
    # look for the
    while found1 == 'false':
        ch1b = min(max(int(ch1b + ch2b), 0), max1)
        if id1 >= sorted_list[sorted_list.__len__() - 1][column_ref]:  # if the value belongs at the end of the list:
            if id1 == sorted_list[sorted_list.__len__() - 1][column_ref]:  # if the value is found at the end of the list
                found1 = 'true'
                break
            else:
                found1 = 'end'
                break
        if sorted_list[ch1b][column_ref] == id1:  # if value is found:
            found1 = 'true'
            break
        else:
            if sorted_list[ch1b][column_ref] > id1:  # if the search area is higher than the search value
                if ch2b < 0:
                    ch2b = round(ch2b / 2)
                else:
                    ch2b = round((ch2b / 2) * -1)
            else:  # the search area is lower than the search value
                if ch2b < 0:
                    ch2b = round((ch2b / 2) * -1)
                else:
                    ch2b = round(ch2b / 2)
            # the search has finished prematurely and needs to proceed incrementally
            if ch2b == 0:
                if sorted_list[min(ch1b + 1, max1)][column_ref] <= id1:
                    ch2b = 1
                else:
                    if sorted_list[max(ch1b - 1, 0)][column_ref] >= id1: # < sorted_list[min(ch1b, max1)][column_ref]:
                        ch2b = -1
        if chkb == ch1b:  # if value is not found:
            # new bit written in Capri
            if ch1b >= sorted_list.__len__() and sorted_list[ch1b][column_ref] < id1:  # if the value belongs at the end
                found1 = 'end'
                break
            else:
                if sorted_list[min(ch1b + 1, max1)][column_ref] == id1 or sorted_list[ch1b - 1][column_ref] == id1:
                    found1 = 'true'
                    break
                else:
                    if sorted_list[ch1b][column_ref] > id1:
                        if sorted_list[ch1b - 1][column_ref] > id1:
                            found1 = max(ch1b - 1, 0)
                            break
                        else:
                            found1 = ch1b
                            break
                    else:
                        found1 = ch1b + 1
                        break
        chkb = ch1b

    return found1


# Assign unique id's to links:
c1 = 0
for line in link_list:
    line += [c1] + [''] + [''] + [''] + [''] + ['']
    c1 += 1

var1 = []
for line in task_list[1:]:
    var1.append([int(line[14])] + [line[42]] + [line[23]] + [line[6]])

var1 = sorted(var1, key=lambda x: (x[0], x[0]), reverse=False)

c1 = -1
c2 = 0
listnopred = []
listnopred2 = []
listnosuc = []
for l2 in link_list:
    cumES = 0
    cumef = 0
    c1 += 1
    if eucfind(l2[3], si2, 2) != 'true':
        c2 += 1
        con1 = eucreturn(l2[3], list1, 0, 3)
        a_st = eucreturn(l2[3], list1, 0, 5)
        p_st = eucreturn(l2[3], list1, 0, 6)
        edur = l2[5]
        pdur = int(l2[1])
        lag = l2[7]
        if a_st != '':
            cumES = getstartday(con1, a_st, '0', l2[6], pdur, l2[7])
            cumef = cumES + float(edur)
            cumes3 = getstartday(con1, a_st, 0.0, 'PR_SS', 0, 0)
            cumef3 = float(cumes3) + float(pdur)
        else:
            if con1 == 'CS_MSOA' or con1 == 'CS_MSO' or con1 == 'CS_MANDSTART' or con1 == 'CS_MANDFIN' or con1 == 'CS_MEOA' or con1 == 'CS_MEO':
                cumES = getstartday(con1, eucreturn(l2[3], list1, 0, 4), edur, l2[6], pdur, l2[7])
                cumef = cumES + float(edur)
                cumes3 = getstartday(con1, eucreturn(l2[3], list1, 0, 4), 0.0, 'PR_SS', 0, 0)
                cumef3 = float(cumes3) + float(pdur)
            else:
                cumES = getstartday('', p_st, edur, l2[6], pdur, l2[7])
                cumef = cumES + float(edur)
                cumes3 = getstartday('', p_st, 0.0, 'PR_SS', 0, 0)
                cumef3 = float(cumes3) + float(pdur)
            cumES2 = getcumes(l2[6], lag, pdur, edur)
            cumef2 = cumES2 + float(edur)
            if l2[6] == 'PR_FS':
                cumes4 = float(cumES2) - float(pdur) - float(l2[7])
            if l2[6] == 'PR_SS':
                cumes4 = float(cumES2) - float(l2[7])
            if l2[6] == 'PR_FF':
                cumes4 = float(cumef2) - float(l2[7])
            if l2[6] == 'PR_SF':
                cumes4 = float(cumef2) - float(pdur) - float(l2[7])
            cumef4 = float(cumes4) + float(pdur)
            if cumes4 > cumes3:
                cumes3 = cumes4
                cumef3 = cumef4
            if cumES2 > cumES:
                cumES = cumES2
                cumef = cumef2
            listnopred2.append(
                [''] + [''] + [l2[3]] + [l2[3]] + [l2[4]] + [
                    eucreturn(l2[3], list1, 0, 2)] + [''] + [0.0] + [task_list.__len__() + c2] + [cumes3] + [
                    cumef3] + [''] + [''] + [''])
        l2[9] = cumES
        l2[10] = cumef
        # need to create a list of tasks with no predecessors
        listnopred.append(l2)

for line in listnopred2:
    listnopred.append(line)

# check for and remove loops
si2 = []
si3 = []
si2 = sorted(link_list, key=lambda x: (x[3], x[0]), reverse=False)
si3 = [d for d in listnopred]
si3 = sorted(si3, key=lambda x: (x[8], x[0]), reverse=False)
si4 = sorted(listnopred, key=lambda x: (x[10], x[0]), reverse=True)
looper = np.arange(1, 3)


# go through the list of tasks with no predecessors 1 at a time
for wt in looper:
    si3 = [d for d in listnopred]
    counter = 0
    for l3 in si4:
        counter += 1
        c2 = -1
        aid1 = int(l3[2])
        print('Pass ' + str(wt) + ' of 2 - Finding and Removing Circular Logic - ' + str("%.2f" % ((si3.__len__() / int(si2.__len__())) * 100)) + "%   " + str(
            int(si4.__len__()) - counter))
        uid1 = int(l3[8])
        pid1 = int(l3[3])
        s_es = int(l3[9])
        s_ef = int(l3[10])
        predef = int(l3[10])
        predes = int(l3[9])
        pstring = str(l3[11])
        ch1 = int(si2.__len__()) - 1
        max2 = int(si2.__len__()) - 1
        found = 'false'
        ch2 = (ch1 / 2) * -1
        chk = 0
        while found == 'false':
            ch1 = min(max(int(ch1 + ch2), 0), max2)
            # if found:
            if si2[ch1][3] == aid1:
                # do what needs to be done
                ch3 = ch1
                ch4 = aid1
                ct1 = 0
                checker = 0
                chk3 = 'false'
                # look down
                while ch4 == aid1:
                    isloop = 'false'
                    if (ch1 + ct1) < si2.__len__():
                        ch4 = si2[ch1 + ct1][3]
                        ch5 = str(si2[ch1 + ct1][2])
                        if ch4 == aid1:
                            cstring = si2[ch1 + ct1][11]
                            if str(aid1) not in pstring and str(aid1) not in cstring:
                                pstring += "-" + str(aid1)
                                si4.insert(counter, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [si2[ch1 + ct1][2]] + [
                                    si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [si2[ch1 + ct1][5]] + [
                                    si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [si2[ch1 + ct1][8]] + [cumES] + [
                                    cumef] + [pstring])
                            si2[ch1 + ct1][11] = pstring
                            if ch5 in pstring:
                                isloop = 'true'
                                si2.pop(ch1 + ct1)
                            if isloop == 'false':
                                cumES = getcumes2(si2[ch1 + ct1][6], si2[ch1 + ct1][7], predes, predef, si2[ch1 + ct1][5])
                                cumef = cumES + int(si2[ch1 + ct1][5])
                                if si2[ch1 + ct1][9] == '':
                                    si2[ch1 + ct1][9] = cumES
                                    si2[ch1 + ct1][10] = cumef
                                else:
                                    if si2[ch1 + ct1][9] < cumES:
                                        si2[ch1 + ct1][9] = cumES
                                        si2[ch1 + ct1][10] = cumef
                                    else:
                                        cumES = si2[ch1 + ct1][9]
                                        cumef = si2[ch1 + ct1][10]

                                checker = eucfind(si2[ch1 + ct1][8], si3, 8)
                                if checker != 'true':
                                    if checker != 'end':
                                        si3.insert(checker, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [
                                            si2[ch1 + ct1][2]] + [si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [
                                            si2[ch1 + ct1][5]] + [si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [
                                            si2[ch1 + ct1][8]] + [cumES] + [cumef] + [pstring])
                                        si4.insert(counter, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [
                                            si2[ch1 + ct1][2]] + [si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [
                                            si2[ch1 + ct1][5]] + [si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [
                                            si2[ch1 + ct1][8]] + [cumES] + [cumef] + [pstring])
                                    else:
                                        si3.append([si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [si2[ch1 + ct1][2]] + [
                                            si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [si2[ch1 + ct1][5]] + [
                                            si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [si2[ch1 + ct1][8]] + [
                                            cumES] + [cumef] + [pstring])
                                        si4.insert(counter, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [
                                            si2[ch1 + ct1][2]] + [si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [
                                            si2[ch1 + ct1][5]] + [si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [
                                            si2[ch1 + ct1][8]] + [cumES] + [cumef] + [pstring])
                    else:
                        break
                    ct1 += 1
                #lookup
                ct1 = -1
                ch4 = aid1
                checker = 0
                while ch4 == aid1:
                    isloop = 'false'
                    if (ch1 + ct1) >= 0:
                        ch4 = si2[ch1 + ct1][3]
                        ch5 = str(si2[ch1 + ct1][2])
                        if ch4 == aid1:
                            cstring = si2[ch1 + ct1][11]
                            if str(aid1) not in pstring and str(aid1) not in cstring:
                                pstring += "-" + str(aid1)
                                si4.insert(counter, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [si2[ch1 + ct1][2]] + [
                                    si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [si2[ch1 + ct1][5]] + [
                                               si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [si2[ch1 + ct1][8]] + [
                                               cumES] + [cumef] + [pstring])
                            si2[ch1 + ct1][11] = pstring
                            if ch5 in pstring:
                                isloop = 'true'
                                si2.pop(ch1 + ct1)
                            if isloop == 'false':
                                cumES = getcumes2(si2[ch1 + ct1][6], si2[ch1 + ct1][7], predes, predef, si2[ch1 + ct1][5])
                                cumef = cumES + int(si2[ch1 + ct1][5])
                                if si2[ch1 + ct1][9] == '':
                                    si2[ch1 + ct1][9] = cumES
                                    si2[ch1 + ct1][10] = cumef
                                else:
                                    if si2[ch1 + ct1][9] < cumES:
                                        si2[ch1 + ct1][9] = cumES
                                        si2[ch1 + ct1][10] = cumef
                                    else:
                                        cumES = si2[ch1 + ct1][9]
                                        cumef = si2[ch1 + ct1][10]

                                checker = eucfind(si2[ch1 + ct1][8], si3, 8)
                                if checker != 'true':
                                    if checker != 'end':
                                        si3.insert(checker, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [
                                            si2[ch1 + ct1][2]] + [si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [
                                            si2[ch1 + ct1][5]] + [si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [
                                            si2[ch1 + ct1][8]] + [cumES] + [cumef] + [pstring])
                                        si4.insert(counter, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [
                                            si2[ch1 + ct1][2]] + [si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [
                                            si2[ch1 + ct1][5]] + [si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [
                                            si2[ch1 + ct1][8]] + [cumES] + [cumef] + [pstring])
                                    else:
                                        si3.append([si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [si2[ch1 + ct1][2]] + [
                                            si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [si2[ch1 + ct1][5]] + [
                                            si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [si2[ch1 + ct1][8]] + [
                                            cumES] + [cumef] + [pstring])
                                        si4.insert(counter, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [
                                            si2[ch1 + ct1][2]] + [si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [
                                            si2[ch1 + ct1][5]] + [si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [
                                            si2[ch1 + ct1][8]] + [cumES] + [cumef] + [pstring])
                    else:
                        break
                    ct1 -= 1
                found = 'true'
                break
            else:
                if si2[ch1][3] > aid1:
                    if ch2 < 0:
                        ch2 = round(ch2 / 2)
                    else:
                        ch2 = round((ch2 / 2) * -1)
                else:
                    if ch2 < 0:
                        ch2 = round((ch2 / 2) * -1)
                    else:
                        ch2 = round(ch2 / 2)
                if ch2 == 0:
                    if si2[min(ch1 + 1, max2)][3] <= aid1:
                        ch2 = 1
                    else:
                        if si2[max(ch1 - 1, 0)][3] >= aid1:
                            ch2 = -1
            if chk == ch1:
                break
            chk = ch1

list2 = []
c1 = 0
sumflt = 0
for line in task_list[1:]:
    list2.append([int(line[14])] + [line[6]] + [line[45]] + [line[46]] + [line[47]] + [line[48]] + [line[49]] + [line[50]] + [line[52]] + [line[42]] + [line[44]] + [line[34]])
    c1 += 1
    sumflt += float(line[34])

list2 = sorted(list2, key=lambda x: (x[0], x[0]), reverse=False)

avflt = sumflt / c1

copied = 'false'
for n in niterng:
    si2 = sorted(si2, key=lambda x: (x[2], x[0]), reverse=False)
    print('Pass ' + str(n) + ' of ' + str(niter))
    c1 = -1
    c2 = 0
    listnopred = []
    listnopred2 = []
    listnosuc = []
    for l2 in link_list:
        cumES = 0
        cumef = 0
        c1 += 1
        if eucfind(l2[3], si2, 2) != 'true':
            c2 += 1
            con1 = eucreturn(l2[3], list1, 0, 3)
            a_st = eucreturn(l2[3], list1, 0, 5)
            p_st = eucreturn(l2[3], list1, 0, 6)
            edur = l2[5]
            pdur = int(l2[1])
            pdurf = 0
            edurf = 0
            # get details for task with no preds
            r1a = region
            w1a = getweatherbooleans(eucreturn(l2[3], list2, 0, 2), 1, r1a)
            w2a = getweatherbooleans(eucreturn(l2[3], list2, 0, 3), 2, r1a)
            w3a = getweatherbooleans(eucreturn(l2[3], list2, 0, 4), 3, r1a)
            w4a = getweatherbooleans(eucreturn(l2[3], list2, 0, 5), 4, r1a)
            w5a = getweatherbooleans(eucreturn(l2[3], list2, 0, 6), 5, r1a)
            w6a = getweatherbooleans(eucreturn(l2[3], list2, 0, 7), 6, r1a)
            m1a = getmarketbool(eucreturn(l2[3], list2, 0, 8))
            conc1a = getconbool(eucreturn(l2[3], list2, 0, 9))
            flt1a = eucreturn(l2[3], list2, 0, 11)
            if flt1a == '':
                flt1a = 0.0
            else:
                flt1a = float(flt1a)
            if flt1a >= avflt:
                flt1a = 0
            else:
                flt1a = 1
            tba = 1
            isdelayeda = bayestrigger(int(l2[0]), int(l2[0]), m1a, flt1a, conc1a, tba, w1a, w2a, w3a, w4a, w5a, w6a)
            if isdelayeda == 'True':
                pdurf = calc_duration_growth([eucreturn(l2[3], var1, 0, 1), 0.0, pdur, eucreturn(l2[3], var1, 0, 3)])
                # ignore durations over 6 months
                if pdur < 130:
                    pdur = pdur * (1 + pdurf)
            else:
                pdurf = 0.0
            # get details for next task
            r1a = region
            w1a = getweatherbooleans(eucreturn(l2[2], list2, 0, 2), 1, r1a)
            w2a = getweatherbooleans(eucreturn(l2[2], list2, 0, 3), 2, r1a)
            w3a = getweatherbooleans(eucreturn(l2[2], list2, 0, 4), 3, r1a)
            w4a = getweatherbooleans(eucreturn(l2[2], list2, 0, 5), 4, r1a)
            w5a = getweatherbooleans(eucreturn(l2[2], list2, 0, 6), 5, r1a)
            w6a = getweatherbooleans(eucreturn(l2[2], list2, 0, 7), 6, r1a)
            m1a = getmarketbool(eucreturn(l2[2], list2, 0, 8))
            conc1a = getconbool(eucreturn(l2[2], list2, 0, 9))
            flt1a = eucreturn(l2[2], list2, 0, 11)
            if flt1a == '':
                flt1a = 0.0
            else:
                flt1a = float(flt1a)
            if flt1a >= avflt:
                flt1a = 0
            else:
                flt1a = 1
            tba = 1
            isdelayeda = bayestrigger(int(eucreturn(l2[2], list2, 0, 10)), int(l2[0]), m1a, flt1a, conc1a, tba, w1a, w2a, w3a, w4a, w5a, w6a)
            if isdelayeda == 'True':
                edurf = calc_duration_growth([eucreturn(l2[2], var1, 0, 1), pdurf, edur, eucreturn(l2[2], var1, 0, 3)])
                # ignore tasks longer than 6 months
                if float(edur) < 130:
                    edur = float(edur) * (1 + edurf)
            lag = l2[7]
            if a_st != '':
                cumES = getstartday(con1, a_st, '0', l2[6], pdur, l2[7])
                cumef = cumES + float(edur)
                cumes3 = getstartday(con1, a_st, 0.0, 'PR_SS', 0, 0)
                cumef3 = float(cumes3) + float(pdur)
            else:
                if con1 == 'CS_MSOA' or con1 == 'CS_MSO' or con1 == 'CS_MANDSTART' or con1 == 'CS_MANDFIN' or con1 == 'CS_MEOA' or con1 == 'CS_MEO':
                    cumES = getstartday(con1, eucreturn(l2[3], list1, 0, 4), edur, l2[6], pdur, l2[7])
                    cumef = cumES + float(edur)
                    cumes3 = getstartday(con1, eucreturn(l2[3], list1, 0, 4), 0.0, 'PR_SS', 0, 0)
                    cumef3 = float(cumes3) + float(pdur)
                else:
                    cumES = getstartday('', p_st, edur, l2[6], pdur, l2[7])
                    cumef = cumES + float(edur)
                    cumes3 = getstartday('', p_st, 0.0, 'PR_SS', 0, 0)
                    cumef3 = float(cumes3) + float(pdur)
                cumES2 = getcumes(l2[6], lag, pdur, edur)
                cumef2 = cumES2 + float(edur)
                if l2[6] == 'PR_FS':
                    cumes4 = float(cumES2) - float(pdur) - float(l2[7])
                if l2[6] == 'PR_SS':
                    cumes4 = float(cumES2) - float(l2[7])
                if l2[6] == 'PR_FF':
                    cumes4 = float(cumef2) - float(l2[7])
                if l2[6] == 'PR_SF':
                    cumes4 = float(cumef2) - float(pdur) - float(l2[7])
                cumef4 = float(cumes4) + float(pdur)
                if cumes4 > cumes3:
                    cumes3 = cumes4
                    cumef3 = cumef4
                if cumES2 > cumES:
                    cumES = cumES2
                    cumef = cumef2
                listnopred2.append(
                    [''] + [''] + [l2[3]] + [l2[3]] + [l2[4]] + [
                        eucreturn(l2[3], list1, 0, 2)] + [''] + [0.0] + [task_list.__len__() + c2] + [cumes3] + [
                        cumef3] + [''] + [''] + [pdurf])
            l2[9] = cumES
            l2[10] = cumef
            l2[13] = edurf
            # need to create a list of tasks with no predecessors
            listnopred.append(l2)

    for line in listnopred2:
        listnopred.append(line)

    si4 = copy.deepcopy(listnopred)
    si3 = copy.deepcopy(listnopred)

    c1 = 0
    for l9 in si2:
        si2[c1][9] = ''
        si2[c1][10] = ''
        si2[c1][11] = ''
        c1 += 1

    si2 = sorted(si2, key=lambda x: (x[3], x[0]), reverse=False)
    si3 = sorted(si3, key=lambda x: (x[8], x[0]), reverse=False)
    si4 = sorted(si4, key=lambda x: (x[10], x[0]), reverse=True)

    # go through the list of tasks with no predecessors 1 at a time
    counter = 0
    for l3 in si4:
        counter += 1
        c2 = -1
        aid1 = int(l3[2])
        uid1 = int(l3[8])
        pid1 = int(l3[3])
        s_es = int(l3[9])
        s_ef = int(l3[10])
        predef = int(l3[10])
        predes = int(l3[9])
        if l3[13] == '':
            pdurf = 0.0
        else:
            pdurf = float(l3[13])
        if pdurf == float(0):
            tba = 0
        else:
            tba = 1
        ch1 = int(si2.__len__()) - 1
        max2 = int(si2.__len__()) - 1
        found = 'false'
        ch2 = (ch1 / 2) * -1
        chk = 0
        while found == 'false':
            ch1 = min(max(int(ch1 + ch2), 0), max2)
            # if found:
            if si2[ch1][3] == aid1:
                # do what needs to be done
                ch3 = ch1
                ch4 = aid1
                ct1 = 0
                checker = 0
                chk3 = 'false'
                # look down
                while ch4 == aid1:
                    if (ch1 + ct1) < si2.__len__():
                        ch4 = si2[ch1 + ct1][3]
                        if ch4 == aid1:
                            edur = si2[ch1 + ct1][5]
                            if si2[ch1 + ct1][13] == '':
                                r1a = region
                                w1a = getweatherbooleans(eucreturn(si2[ch1 + ct1][2], list2, 0, 2), 1, r1a)
                                w2a = getweatherbooleans(eucreturn(si2[ch1 + ct1][2], list2, 0, 3), 2, r1a)
                                w3a = getweatherbooleans(eucreturn(si2[ch1 + ct1][2], list2, 0, 4), 3, r1a)
                                w4a = getweatherbooleans(eucreturn(si2[ch1 + ct1][2], list2, 0, 5), 4, r1a)
                                w5a = getweatherbooleans(eucreturn(si2[ch1 + ct1][2], list2, 0, 6), 5, r1a)
                                w6a = getweatherbooleans(eucreturn(si2[ch1 + ct1][2], list2, 0, 7), 6, r1a)
                                m1a = getmarketbool(eucreturn(si2[ch1 + ct1][2], list2, 0, 8))
                                conc1a = getconbool(eucreturn(si2[ch1 + ct1][2], list2, 0, 9))
                                flt1a = eucreturn(si2[ch1 + ct1][2], list2, 0, 11)
                                if flt1a == '':
                                    flt1a = 0.0
                                else:
                                    flt1a = float(flt1a)
                                if flt1a >= avflt:
                                    flt1a = 0
                                else:
                                    flt1a = 1
                                isdelayeda = bayestrigger(int(eucreturn(si2[ch1 + ct1][2], list2, 0, 10)), int(si2[ch1 + ct1][0]), m1a, flt1a, conc1a,
                                                          tba, w1a, w2a, w3a, w4a, w5a, w6a)
                                if isdelayeda == 'True':
                                    edurf = calc_duration_growth([eucreturn(si2[ch1 + ct1][2], var1, 0, 1), pdurf, edur, eucreturn(si2[ch1 + ct1][2], var1, 0, 3)])
                                    if float(edur) < 130:
                                        edur = float(edur) * (1 + edurf)
                                    si2[ch1 + ct1][5] = edur
                                else:
                                    edurf = 0.0
                            else:
                                edurf = si2[ch1 + ct1][13]
                            cumES = getcumes2(si2[ch1 + ct1][6], si2[ch1 + ct1][7], predes, predef, edur)
                            cumef = cumES + float(edur)
                            if si2[ch1 + ct1][9] == '':
                                si2[ch1 + ct1][9] = cumES
                                si2[ch1 + ct1][10] = cumef
                                si2[ch1 + ct1][13] = edurf
                            else:
                                if si2[ch1 + ct1][10] < cumef:
                                    si2[ch1 + ct1][9] = cumES
                                    si2[ch1 + ct1][10] = cumef
                                    si2[ch1 + ct1][13] = edurf
                                    si4.insert(counter, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [si2[ch1 + ct1][2]] + [
                                        si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [si2[ch1 + ct1][5]] + [
                                                   si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [si2[ch1 + ct1][8]] + [
                                                   cumES] + [cumef] + [pstring] + [''] + [edurf])
                                else:
                                    cumES = si2[ch1 + ct1][9]
                                    cumef = si2[ch1 + ct1][10]

                            checker = eucfind(si2[ch1 + ct1][8], si3, 8)
                            if checker != 'true':
                                if checker != 'end':
                                    si3.insert(checker, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [
                                        si2[ch1 + ct1][2]] + [si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [
                                                   si2[ch1 + ct1][5]] + [si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [
                                                   si2[ch1 + ct1][8]] + [cumES] + [cumef] + [pstring] + [''] + [edurf])
                                    si4.insert(counter, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [
                                        si2[ch1 + ct1][2]] + [si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [
                                                   si2[ch1 + ct1][5]] + [si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [
                                                   si2[ch1 + ct1][8]] + [cumES] + [cumef] + [pstring] + [''] + [edurf])
                                else:
                                    si3.append([si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [si2[ch1 + ct1][2]] + [
                                        si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [si2[ch1 + ct1][5]] + [
                                                   si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [si2[ch1 + ct1][8]] + [
                                                   cumES] + [cumef] + [pstring] + [''] + [edurf])
                                    si4.insert(counter, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [
                                        si2[ch1 + ct1][2]] + [si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [
                                                   si2[ch1 + ct1][5]] + [si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [
                                                   si2[ch1 + ct1][8]] + [cumES] + [cumef] + [pstring] + [''] + [edurf])
                    else:
                        break
                    ct1 += 1
                #lookup
                ct1 = -1
                ch4 = aid1
                checker = 0
                while ch4 == aid1:
                    if (ch1 + ct1) >= 0:
                        ch4 = si2[ch1 + ct1][3]
                        if ch4 == aid1:
                            edur = si2[ch1 + ct1][5]
                            if si2[ch1 + ct1][13] == '':
                                r1a = region
                                w1a = getweatherbooleans(eucreturn(si2[ch1 + ct1][2], list2, 0, 2), 1, r1a)
                                w2a = getweatherbooleans(eucreturn(si2[ch1 + ct1][2], list2, 0, 3), 2, r1a)
                                w3a = getweatherbooleans(eucreturn(si2[ch1 + ct1][2], list2, 0, 4), 3, r1a)
                                w4a = getweatherbooleans(eucreturn(si2[ch1 + ct1][2], list2, 0, 5), 4, r1a)
                                w5a = getweatherbooleans(eucreturn(si2[ch1 + ct1][2], list2, 0, 6), 5, r1a)
                                w6a = getweatherbooleans(eucreturn(si2[ch1 + ct1][2], list2, 0, 7), 6, r1a)
                                m1a = getmarketbool(eucreturn(si2[ch1 + ct1][2], list2, 0, 8))
                                conc1a = getconbool(eucreturn(si2[ch1 + ct1][2], list2, 0, 9))
                                flt1a = eucreturn(si2[ch1 + ct1][2], list2, 0, 11)
                                if flt1a == '':
                                    flt1a = 0.0
                                else:
                                    flt1a = float(flt1a)
                                if flt1a >= avflt:
                                    flt1a = 0
                                else:
                                    flt1a = 1
                                isdelayeda = bayestrigger(int(eucreturn(si2[ch1 + ct1][2], list2, 0, 10)), int(si2[ch1 + ct1][0]), m1a, flt1a, conc1a,
                                                          tba, w1a, w2a, w3a, w4a, w5a, w6a)
                                if isdelayeda == 'True':
                                    edurf = calc_duration_growth([eucreturn(si2[ch1 + ct1][2], var1, 0, 1), pdurf, edur, eucreturn(si2[ch1 + ct1][2], var1, 0, 3)])
                                    if float(edur) < 130:
                                        edur = float(edur) * (1 + edurf)
                                    si2[ch1 + ct1][5] = edur
                                else:
                                    edurf = 0.0
                            else:
                                edurf = si2[ch1 + ct1][13]
                            cumES = getcumes2(si2[ch1 + ct1][6], si2[ch1 + ct1][7], predes, predef, edur)
                            cumef = cumES + float(edur)
                            if si2[ch1 + ct1][9] == '':
                                si2[ch1 + ct1][9] = cumES
                                si2[ch1 + ct1][10] = cumef
                                si2[ch1 + ct1][13] = edurf
                            else:
                                if si2[ch1 + ct1][10] < cumef:
                                    si2[ch1 + ct1][9] = cumES
                                    si2[ch1 + ct1][10] = cumef
                                    si2[ch1 + ct1][13] = edurf
                                    si4.insert(counter, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [si2[ch1 + ct1][2]] + [
                                        si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [si2[ch1 + ct1][5]] + [
                                                   si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [si2[ch1 + ct1][8]] + [
                                                   cumES] + [cumef] + [pstring] + [''] + [edurf])
                                else:
                                    cumES = si2[ch1 + ct1][9]
                                    cumef = si2[ch1 + ct1][10]

                            checker = eucfind(si2[ch1 + ct1][8], si3, 8)
                            if checker != 'true':
                                if checker != 'end':
                                    si3.insert(checker, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [
                                        si2[ch1 + ct1][2]] + [si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [
                                                   si2[ch1 + ct1][5]] + [si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [
                                                   si2[ch1 + ct1][8]] + [cumES] + [cumef] + [pstring] + [''] + [edurf])
                                    si4.insert(counter, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [
                                        si2[ch1 + ct1][2]] + [si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [
                                                   si2[ch1 + ct1][5]] + [si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [
                                                   si2[ch1 + ct1][8]] + [cumES] + [cumef] + [pstring] + [''] + [edurf])
                                else:
                                    si3.append([si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [si2[ch1 + ct1][2]] + [
                                        si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [si2[ch1 + ct1][5]] + [
                                                   si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [si2[ch1 + ct1][8]] + [
                                                   cumES] + [cumef] + [pstring] + [''] + [edurf])
                                    si4.insert(counter, [si2[ch1 + ct1][0]] + [si2[ch1 + ct1][1]] + [
                                        si2[ch1 + ct1][2]] + [si2[ch1 + ct1][3]] + [si2[ch1 + ct1][4]] + [
                                                   si2[ch1 + ct1][5]] + [si2[ch1 + ct1][6]] + [si2[ch1 + ct1][7]] + [
                                                   si2[ch1 + ct1][8]] + [cumES] + [cumef] + [pstring] + [''] + [edurf])
                    else:
                        break
                    ct1 -= 1
                found = 'true'
                break
            else:
                if si2[ch1][3] > aid1:
                    if ch2 < 0:
                        ch2 = round(ch2 / 2)
                    else:
                        ch2 = round((ch2 / 2) * -1)
                else:
                    if ch2 < 0:
                        ch2 = round((ch2 / 2) * -1)
                    else:
                        ch2 = round(ch2 / 2)
                if ch2 == 0:
                    if si2[min(ch1 + 1, max2)][3] <= aid1:
                        ch2 = 1
                    else:
                        if si2[max(ch1 - 1, 0)][3] >= aid1:
                            ch2 = -1
            if chk == ch1:
                #print(aid1)
                if listnosuc.__len__() != 0:
                    if listnosuc[max(listnosuc.__len__() - 1, 0)][2] != aid1:
                        listnosuc.append(l3)
                else:
                    listnosuc.append(l3)
                break
            chk = ch1

    # empty old list from the memory
    si3 = []
    si4 = []

    si5 = copy.deepcopy(si2)
    for l9 in si5:
        l9[11] = ''
        l9 += ['']

    for line in listnopred2:
        si5.append(line)

    si5 = sorted(si5, key=lambda x: (x[2], x[0]), reverse=False)

    # work out the ALAP dates
    # create a list of tasks only giving their early dates
    c1 = 0
    previd = 'random nr'
    for l9 in si5:
        actid = l9[2]
        if actid != previd:
            if l9[9] == '':
                s_es = 0
                s_ef = int(l9[5])
            else:
                s_es = l9[9]
                s_ef = l9[10]
            ch5 = actid
            c2 = 0
            while ch5 == actid:
                if c1 + c2 == si5.__len__():
                    si3.append([int(actid)] + [s_es] + [s_ef])
                    break
                else:
                    ch5 = si5[c1 + c2][2]
                    if ch5 == actid:
                        if si5[c1 + c2][10] == '':
                            si5[c1 + c2][9] = 0
                            si5[c1 + c2][10] = int(si5[c1 + c2][5])
                        if si5[c1 + c2][10] > s_ef:
                            s_ef = si5[c1 + c2][10]
                            s_es = si5[c1 + c2][9]
                    else:
                        si3.append([int(actid)] + [s_es] + [s_ef])
                        break
                    c2 += 1
        previd = actid
        c1 += 1

    si6 = sorted(listnosuc, key=lambda x: (x[2], x[0]), reverse=False)
    si7 = []
    si7 = [d for d in si6]
    previd = 'rand'
    tlist = []
    for l9 in si7:
        actid = l9[2]
        if actid != previd:
            ls = eucreturn(actid, si3, 0, 1)
            lf = eucreturn(actid, si3, 0, 2)
            tlist.append([actid] + [ls] + [lf] + [0] + [0])
            previd = actid

    si6 = []
    si6 = [d for d in tlist]
    for l9 in tlist:
        # go through the list of tasks with no successors 1 at a time
        c2 = -1
        aid1 = l9[0]
        ls = l9[1]
        lf = l9[2]
        ch1 = int(si5.__len__()) - 1
        max2 = int(si5.__len__()) - 1
        found = 'false'
        ch2 = (ch1 / 2) * -1
        chk = 0
        while found == 'false':
            ch1 = min(max(int(ch1 + ch2), 0), max2)
            # if found:
            if int(si5[ch1][2]) == aid1:
                # do what needs to be done
                ch3 = ch1
                ch4 = aid1
                ct1 = 0
                checker = 0
                # look down
                while ch4 == aid1:
                    if (ch1 + ct1) < si5.__len__():
                        ch4 = int(si5[ch1 + ct1][2])
                        if ch4 == aid1:
                            if si5[ch1 + ct1][11] == '':
                                si5[ch1 + ct1][11] = ls
                                si5[ch1 + ct1][12] = lf
                            else:
                                if si5[ch1 + ct1][11] > ls:
                                    si5[ch1 + ct1][11] = ls
                                    si5[ch1 + ct1][12] = lf
                                    tlist.append([si5[ch1 + ct1][3]] + [
                                        getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]), si5[ch1 + ct1][1], float(si5[ch1 + ct1][5])) - float(si5[ch1 + ct1][5])] + [
                                                     getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]),
                                                              si5[ch1 + ct1][1], int(si5[ch1 + ct1][5]))])
                                else:
                                    ls = si5[ch1 + ct1][11]
                                    lf = si5[ch1 + ct1][12]

                            checker = eucfind(si5[ch1 + ct1][3], si6, 0)
                            if checker != 'true':
                                if checker != 'end':
                                    tlist.append([si5[ch1 + ct1][3]] + [
                                        getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]), si5[ch1 + ct1][1], float(si5[ch1 + ct1][5])) - float(si5[ch1 + ct1][5])] + [
                                                     getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]),
                                                              si5[ch1 + ct1][1], int(si5[ch1 + ct1][5]))])
                                    si6.insert(max(0, checker), [si5[ch1 + ct1][3]] + [
                                        getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]), si5[ch1 + ct1][1], float(si5[ch1 + ct1][5])) - float(si5[ch1 + ct1][5])] + [
                                                     getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]),
                                                              si5[ch1 + ct1][1], int(si5[ch1 + ct1][5]))])
                                else:
                                    tlist.append([si5[ch1 + ct1][3]] + [
                                        getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]), si5[ch1 + ct1][1], float(si5[ch1 + ct1][5])) - float(si5[ch1 + ct1][5])] + [
                                                     getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]),
                                                              si5[ch1 + ct1][1], int(si5[ch1 + ct1][5]))])
                                    si6.append([si5[ch1 + ct1][3]] + [
                                        getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]), si5[ch1 + ct1][1], float(si5[ch1 + ct1][5])) - float(si5[ch1 + ct1][5])] + [
                                                     getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]),
                                                              si5[ch1 + ct1][1], int(si5[ch1 + ct1][5]))])
                    else:
                        break
                    ct1 += 1
                # lookup
                ct1 = -1
                ch4 = aid1
                checker = 0
                while ch4 == aid1:
                    if (ch1 + ct1) < si5.__len__():
                        ch4 = int(si5[ch1 + ct1][2])
                        if ch4 == aid1:
                            if si5[ch1 + ct1][11] == '':
                                si5[ch1 + ct1][11] = ls
                                si5[ch1 + ct1][12] = lf
                            else:
                                if si5[ch1 + ct1][11] > ls:
                                    si5[ch1 + ct1][11] = ls
                                    si5[ch1 + ct1][12] = lf
                                    tlist.append([si5[ch1 + ct1][3]] + [
                                        getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]), si5[ch1 + ct1][1], float(si5[ch1 + ct1][5])) - float(si5[ch1 + ct1][5])] + [
                                                     getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]),
                                                              si5[ch1 + ct1][1], int(si5[ch1 + ct1][5]))])
                                else:
                                    ls = si5[ch1 + ct1][11]
                                    lf = si5[ch1 + ct1][12]

                            checker = eucfind(si5[ch1 + ct1][3], si6, 0)
                            if checker != 'true':
                                if checker != 'end':
                                    tlist.append([si5[ch1 + ct1][3]] + [
                                        getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]), si5[ch1 + ct1][1], float(si5[ch1 + ct1][5])) - float(si5[ch1 + ct1][5])] + [
                                                     getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]),
                                                              si5[ch1 + ct1][1], int(si5[ch1 + ct1][5]))])
                                    si6.insert(max(0, checker), [si5[ch1 + ct1][3]] + [
                                        getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]), si5[ch1 + ct1][1], float(si5[ch1 + ct1][5])) - float(si5[ch1 + ct1][5])] + [
                                                     getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]),
                                                              si5[ch1 + ct1][1], int(si5[ch1 + ct1][5]))])
                                else:
                                    tlist.append([si5[ch1 + ct1][3]] + [
                                        getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]), si5[ch1 + ct1][1], float(si5[ch1 + ct1][5])) - float(si5[ch1 + ct1][5])] + [
                                                     getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]),
                                                              si5[ch1 + ct1][1], int(si5[ch1 + ct1][5]))])
                                    si6.append([si5[ch1 + ct1][3]] + [
                                        getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]), si5[ch1 + ct1][1], float(si5[ch1 + ct1][5])) - float(si5[ch1 + ct1][5])] + [
                                                     getcumlf(si5[ch1 + ct1][6], ls, lf, float(si5[ch1 + ct1][7]),
                                                              si5[ch1 + ct1][1], int(si5[ch1 + ct1][5]))])
                    else:
                        break
                    ct1 -= 1
                found = 'true'
                break
            else:
                if int(si5[ch1][2]) > aid1:
                    if ch2 < 0:
                        ch2 = round(ch2 / 2)
                    else:
                        ch2 = round((ch2 / 2) * -1)
                else:
                    if ch2 < 0:
                        ch2 = round((ch2 / 2) * -1)
                    else:
                        ch2 = round(ch2 / 2)
                if ch2 == 0:
                    if int(si5[min(ch1 + 1, max2)][2]) <= aid1:
                        ch2 = 1
                    else:
                        if int(si5[max(ch1 - 1, 0)][2]) >= aid1:
                            ch2 = -1
            if chk == ch1:
                break
            chk = ch1

    for line8 in si5:
        if line8[11] == '':
            line8[11] = line8[9]
        if line8[12] == '':
            line8[12] = line8[10]

    if copied == 'false':
        copylist = copy.deepcopy(si5)
        copied = 'true'
    else:
        cc1 = 0
        for line in copylist:
            line[9] = ((line[9] * (n - 1)) + si5[cc1][9]) / n
            line[10] = ((line[10] * (n - 1)) + si5[cc1][10]) / n
            line[11] = ((line[11] * (n - 1)) + si5[cc1][11]) / n
            line[12] = ((line[12] * (n - 1)) + si5[cc1][12]) / n
            cc1 += 1

# create a list of tasks only giving their early dates
c1 = 0
si3 = []
previd = 'random nr'
for l9 in copylist:
    actid = l9[2]
    if actid != previd:
        if l9[9] == '':
            s_es = 0
            s_ef = int(l9[5])
        else:
            s_es = l9[9]
            s_ef = l9[10]
        ch5 = actid
        c2 = 0
        while ch5 == actid:
            if c1 + c2 == copylist.__len__():
                si3.append([int(actid)] + [s_es] + [s_ef])
                break
            else:
                ch5 = copylist[c1 + c2][2]
                if ch5 == actid:
                    if copylist[c1 + c2][10] == '':
                        copylist[c1 + c2][9] = 0
                        copylist[c1 + c2][10] = int(copylist[c1 + c2][5])
                    if copylist[c1 + c2][10] > s_ef:
                        s_ef = copylist[c1 + c2][10]
                        s_es = copylist[c1 + c2][9]
                else:
                    si3.append([int(actid)] + [s_es] + [s_ef])
                    break
                c2 += 1
    previd = actid
    c1 += 1

final_list = []
for line in task_list[1:]:
    ch1 = eucreturn(int(line[14]), copylist, 2, 12)
    if ch1 == '#N/A' or ch1 == '':
        con1 = line[35]
        cd1 = line[36]
        a_st = line[28]
        edur = line[23]
        if a_st != '':
            cumES = getstartday('', a_st, edur, 'PR_SS', 0.0, 0.0)
        else:
            if con1 == 'CS_MSOA' or con1 == 'CS_MSO' or con1 == 'CS_MANDSTART' or con1 == 'CS_MANDFIN' or con1 == 'CS_MEOA' or con1 == 'CS_MEO':
                cumES = getstartday(con1, cd1, edur, 'PR_SS', 0.0, 0.0)
            else:
                cumES = 0
        cumef = float(cumES) + float(edur)
        final_list.append([line[44]] + [line[14]] + [line[16]] + [line[17].replace(',', '')] + [cumES] + [cumef] + [0.0])
    else:
        final_list.append([line[44]] + [line[14]] + [line[16]] + [line[17].replace(',', '')] + [eucreturn(int(line[14]), si3, 0, 1)] + [eucreturn(int(line[14]), si3, 0, 2)] + [float(eucreturn(int(line[14]), copylist, 2, 12)) - float(eucreturn(int(line[14]), copylist, 2, 10))])

with open("SimulationOutput2.txt", "a") as output:
    for line in final_list:
        output.write(str(line).replace('[', '').replace(']', '').replace("\r\n", '') + '\n')