import pickle
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as font_manager
import csv
from collections import defaultdict

fontpath = '/Users/Nick/Downloads/Palatino-Roman.ttf'
prop = font_manager.FontProperties(fname=fontpath)
matplotlib.rcParams['font.family'] = prop.get_name()

from pylab import *

oldsavefig = savefig
def savefig(*args, **vargs):
    if "transparent" not in vargs:
        oldsavefig(transparent=True, *args, **vargs)
    else:
        oldsavefig(*args, **vargs)

TT = 8

LIGHT_GRAY = (.6, .6, .6)
DARK_GRAY = (.4, .4, .4)
LIGHTEST_GRAY = (.8, .8, .8)
LIGHT_ORANGE = (1., .7, .4)
DARK_ORANGE = (1., .4, 0.)
LIGHTEST_ORANGE = (1., .9, .7)
PURPLE = (0.6, 0., 0.8)
LIGHT_BLUE = (0.0, 0.8, 1.)
DARK_BLUE = (0.0, 0.4, 1.)
BLACK=(0.0,0.0,0.0)
RED = (1.0,0.0,0.0)
GREEN = (0.0,1.0,0.0)
ORANGE_RED=(1.0,0.27,0.)

dt = 0.5

def lighter(c):
    t = 0.5
    return tuple([c[i]*(1-t)+t*1. for i in range(3)])

def ls(pattern):
    output = subprocess.check_output("ls {}".format(pattern), shell=True).splitlines()
    return output

def load(filename):
    with open(filename) as f:
        ret = pickle.load(f)
    u, x, b = ret
    uh, ur = u
    xh, xr = x
    b = b[1]
    t = arange(len(xh))*dt
    user = int(filename.split('/')[1].split('-')[0][1:])
    distracted = (1 if filename.split('.')[-1]=='distracted' else 0)
    world = filename.split('/')[-1].split('_')[0]
    active = (1 if filename.split('/')[-1].split('_')[1].split('-')[0]=='active' else 0)
    return {
        'uh': asarray(uh), 'ur': asarray(ur), 'xh': asarray(xh), 'xr': asarray(xr), 'b': asarray(b), 't': t,
        'user': user,
        'distracted': distracted,
        'world': world,
        'active': active
    }

def isempty(data):
    return len(data['t'])==0

def extend(a, w):
    if len(a)>=w:
        return a[:w]
    return concatenate([a, nan*ones(w-len(a))])

def cextend(a, w):
    if len(a)>=w:
        return a[:w]
    return concatenate([a, asarray([a[-1]]*(w-len(a)))])

#worlds = ['world{}'.format(i) for i in range(1, 4)]
#datasets = {}
#for w in worlds:
#    datasets[w] = [load(x) for x in ls("data/*/{}*".format(w))]

#for w, dataset in datasets.items():
#    print '{}: {} samples'.format(w, len(dataset))
#print '-'*20

def setup():
    gca().spines['right'].set_visible(False)
    gca().spines['top'].set_visible(False)
    gca().xaxis.set_ticks_position('bottom')
    gca().yaxis.set_ticks_position('left')
    tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

def gen_csvs():
    with open('plots/data.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['trajectory_id', 'time', 'user', 'scenario', 'active', 'distracted', 'belief_attentive', 'belief_distracted'])
        writer.writeheader()
        cnt = 0
        for world in ['world1', 'world2', 'world3']:
            T = max([len(data['t']) for data in datasets[world]])
            for data in datasets[world]:
                if data['user']!=0:
                    cnt += 1
                    for t, b, c in zip(asarray([i*dt for i in range(T)]), cextend(data['b'][:, 0], T), cextend(data['b'][:, 1], T)):
                        writer.writerow({
                            'trajectory_id': cnt,
                            'time': t,
                            'user': data['user'],
                            'scenario': data['world'],
                            'active': data['active'],
                            'distracted': data['distracted'],
                            'belief_attentive': b,
                            'belief_distracted': c
                        })

def plot_world(world):
    T = max([len(data['t']) for data in datasets[world]])
    t = asarray([i*dt for i in range(T)])
    figure()
    plots = []
    for i in range(2):
        subplot(2, 2, i+1)
        ylim(0, 1)
        setup()
        #xlabel('time\n({})'.format(['a', 'b'][i]))
        #ylabel('$b(\\varphi=$attentive$)$')
        for data in datasets[world][::-1]:
            if data['distracted']==i and data['user']!=0:
                plot(t, cextend(data['b'][:, 0], T), color=(LIGHT_ORANGE if data['active']==1 else LIGHT_GRAY))
            if data['distracted']==i and data['user']==0:
                plot(t, cextend(data['b'][:, 0], T), color=(PURPLE if data['active']==1 else LIGHT_BLUE))
    for i in range(2):
        subplot(2, 2, i+3)
        ylim(0, 1)
        setup()
        #xlabel('time\n({})'.format(['c', 'd'][i]))
        #ylabel('$b(\\varphi=$attentive$)$')
        for j in range(2):
            d = np.stack([cextend(data['b'][:, 0], T) for data in datasets[world] if data['distracted']==i and data['active']==j and data['user']!=0])
            m = mean(d, axis=0)
            s = std(d, axis=0)
            N = len(d)
            s = s/sqrt(N)
            fill_between(t, m-s, m+s, color=(LIGHT_ORANGE if j==1 else LIGHT_GRAY))
            plots.append(plot(t, m, color=(DARK_ORANGE if j==1 else DARK_GRAY), linewidth=3.))
            for data in datasets[world]:
                if data['user']==0 and data['distracted']==i and data['active']==j:
                    plots.append(plot(t, cextend(data['b'][:, 0], T), color=(PURPLE if data['active']==1 else LIGHT_BLUE),linewidth=3.))
    figlegend((plots[1][0], plots[0][0], plots[3][0], plots[2][0]), ('Learned Human Model (Passive)', 'Passive Info Gathering', 'Learned Human Model (Active)', 'Active Info Gathering'), 'upper center', ncol=2, fontsize=10, frameon=False)
    savefig('plots/{}.pdf'.format(world))
    if world=='world1':
        figure(figsize=(3, 5))
    else:
        figure(figsize=(9, 5))
    gca().spines['right'].set_visible(False)
    gca().spines['top'].set_visible(False)
    gca().xaxis.set_ticks_position('none')
    gca().yaxis.set_ticks_position('none')
    plots = [None, None]
    for data in datasets[world]:
        if world=='world1':
            axis('equal')
            gcf().subplots_adjust(left=0.2, top=0.85)
            X = data['xr'][:, 0]
            #xlabel('x')
            Y = data['xr'][:, 1]
            #ylabel('y')
        elif world=='world2':
            X = data['t']
            #xlabel('time')
            xlim(0, 15)
            Y = data['xr'][:, 3]
            #ylabel('speed')
        elif world=='world3':
            X = data['t']
            #xlabel('time')
            xlim(0, 10)
            Y = data['xr'][:, 0]
            #ylabel('x')
        plots[data['active']] = plot(X, Y, color=(LIGHT_ORANGE if data['active']==1 else LIGHT_GRAY))
    figlegend((plots[0][0], plots[1][0]), ('Passive Info Gathering', 'Active Info Gathering'), 'upper center', ncol=(2 if world!='world1' else 1), fontsize=12, frameon=False)
    savefig('plots/{}-traj.pdf'.format(world))

def plot4():
    fontsize=20
    def setup():
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plots = []
    figure(figsize=(9, 12))
    for i in range(1, 4):
        world = 'world{}'.format(i)
        T = max([len(data['t']) for data in datasets[world] if data['user']!=0])
        t = asarray([u*dt for u in range(T)])
        for j in range(2):
            subplot(3, 2, 2*i-1+j)
            ylim(-0.1, 1.1)
            setup()
            #xlabel('time\n({}) Scenario {}. {}'.format('abcdefg'[2*i-2+j], i, ['Attentive', 'Distracted'][j]), fontsize=fontsize)
            #ylabel('$b(\\varphi=$attentive$)$', fontsize=fontsize)
            for k in range(2):
                d = np.stack([cextend(data['b'][:, 0], T) for data in datasets[world] if data['distracted']==j and data['active']==k and data['user']!=0])
                m = mean(d, axis=0)
                s = std(d, axis=0)
                N = len(d)
                s = s/sqrt(N)
                if k==1:
                    if j==0:
                        color = ORANGE_RED
                        bcolor = LIGHT_ORANGE
                    else:
                        color = LIGHT_ORANGE
                        bcolor = LIGHTEST_ORANGE
                else:
                    if j==0:
                        color = BLACK
                        bcolor = LIGHT_GRAY
                    else:
                        color = LIGHT_GRAY
                        bcolor = LIGHTEST_GRAY
                fill_between(t, m-s, m+s, color=lighter(color))
                plots.append(plot(t, m, color=color, linewidth=2.))
                for data in datasets[world]:
                    if data['user']==0 and data['distracted']==j and data['active']==k:
                        pass#plots.append(plot(t, cextend(data['b'][:, 0], T), color=color, linewidth=2., linestyle='dashed'))
    #figlegend((plots[1][0], plots[0][0], plots[3][0], plots[2][0]), ('Learned Human Model (Passive)', 'Passive Info Gathering', 'Learned Human Model (Active)', 'Active Info Gathering'), 'upper center', ncol=2, fontsize=12, frameon=False)
    tight_layout()
    savefig('plots/fig4.pdf')

def plot5():
    def setup():
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().xaxis.set_ticks_position('none')
        gca().yaxis.set_ticks_position('none')
    fig1 = figure(figsize=(12, 4))
    fig2 = figure(figsize=(12, 8))
    fig3 = figure(figsize=(4, 4))
    figure(figsize=(12, 10))
    plots = [None, None]
    for w in range(1, 4):
        world = 'world{}'.format(w)
        figure(fig1.number)
        subplot(1, 3, w)
        setup()
        def Xlabel(s):
            xlabel('{}\n({}) Scenario {}. Robot Trajectory (Passive and Active)'.format(s, 'abc'[w-1], w))
        plots = defaultdict(list)
        def plotit(X, Y, color, **style):
            plots[color, tuple(style.items())].append((X, Y))
        for data in datasets[world]:
            if data['user']==0:
                style = {'linestyle': 'dashed', 'linewidth': 2.}
                continue
            else:
                style = {}
            if world=='world1':
                ylim(0, 1.6)
                xlim(-.3, .3)
                X = data['xr'][:, 0]
                #Xlabel('x')
                Y = data['xr'][:, 1]
                #ylabel('y')
            elif world=='world2':
                X = data['t']
                #Xlabel('time')
                xlim(0, TT)
                Y = data['xr'][:, 3]
                #ylabel('speed')
            elif world=='world3':
                X = data['t']
                #Xlabel('time')
                xlim(0, TT)
                Y = data['xr'][:, 0]
                #ylabel('x')
            if data['active']==1:
                if data['distracted']==0:
                    color = ORANGE_RED
                else:
                    color = LIGHT_ORANGE
            else:
                continue
                if data['distracted']==0:
                    color = BLACK
                else:
                    color = LIGHT_GRAY
            plotit(X, Y, color=color, **style)
        def plotall(special=False, clear=True):
            for (color, style), vals in plots.iteritems():
                L = max([len(X) for X, Y in vals])
                L = min(2*TT+1, L)
                def nanfill(a):
                    if len(a)<=L:
                        return list(a)+[nan]*(L-len(a))
                    else:
                        return a[:L]
                theX = np.array([nanfill(X) for X, Y in vals])
                theY = np.array([nanfill(Y) for X, Y in vals])
                mX = nanmean(theX, axis=0)
                sX = nanstd(theX, axis=0)/sqrt(theX.shape[0])
                mY = nanmean(theY, axis=0)
                sY = nanstd(theY, axis=0)/sqrt(theY.shape[0])
                if style==():
                    style = (('linewidth', 2.),)
                plot(mX, mY, color=color, **dict(style))
                if len(vals)==1:
                    continue
                if world=='world1' and special:
                    fill_betweenx(mY, mX-sqrt(sX**2+sY**2), mX+sqrt(sX**2+sY**2), color=lighter(color))
                else:
                    fill_between(mX, mY-sY, mY+sY, color=lighter(color))
            if clear:
                plots.clear()
        if world=='world1':
            plotall(True, False)
            figure(fig3.number)
            axis('equal')
            axis('off')
            plotall(True)
        else:
            plotall(True)
        for i in range(2):
            figure(fig2.number)
            subplot(2, 3, i*3+w)
            setup()
            #xlabel('time\n({}) S{}. {}'.format('abcdefghi'[3*w-3+1+i], w, ['Active', 'Passive'][i]))
            #ylabel('speed')
            def Xlabel(s):
                xlabel('{}\n({}) Scenario {}. Human Trajectory ({})'.format('time', 'abcdef'[3*i+w-1], w, ['Passive', 'Active'][1-i]))
            for data in datasets[world]:
                if data['active']==i:
                    continue
                if data['user']==0:
                    style = {'linestyle': 'dashed', 'linewidth': 2.}
                    continue
                else:
                    style = {}
                if world=='world1' and data['active']==1:
                    ylim(0, 2)
                    xlim(0, TT)
                    X = data['t']
                    #Xlabel('time')
                    Y = abs(data['xh'][:, 0]**2-data['xr'][:,0])+ abs(data['xh'][:,1]-data['xr'][:,1]**2)
                    #ylabel('distance between robot and human')
                    plotit(X, Y, color= (ORANGE_RED if data['distracted']==0 else LIGHT_ORANGE), **style)
                elif world=='world2' and data['active']==1:
                    X = data['t']
                    #Xlabel('time')
                    xlim(0, TT)
                    Y = abs(data['xh'][:, 1]-data['xr'][:, 1])
                    ylim(0, 0.5)
                    #ylabel('lateral distance')
                    plotit(X, Y, color= (ORANGE_RED if data['distracted']==0 else LIGHT_ORANGE), **style)
                elif world=='world3' and data['active']==1:
                    X = data['t']
                    #Xlabel('time')
                    xlim(0, TT)
                    Y = data['xh'][:, 1]
                    ylim(-0.5,0.6)
                    #ylabel('y of human')
                    plotit(X, Y, color= (ORANGE_RED if data['distracted']==0 else LIGHT_ORANGE), **style)
                elif world=='world1' and data['active']==0:
                    ylim(0, 2)
                    xlim(0, TT)
                    X = data['t']
                    #Xlabel('time')
                    Y = abs(data['xh'][:, 0]**2-data['xr'][:,0])+ abs(data['xh'][:,1]-data['xr'][:,1]**2)
                    #ylabel('distance between robot and human')
                    plotit(X, Y, color= (BLACK if data['distracted']==0 else LIGHT_GRAY), **style)
                elif world=='world2' and data['active']==0:
                    X = data['t']
                    #Xlabel('time')
                    xlim(0, TT)
                    Y = abs(data['xh'][:, 1]-data['xr'][:, 1])
                    ylim(0,0.6)
                    #ylabel('lateral distance')
                    plotit(X, Y, color= (BLACK if data['distracted']==0 else LIGHT_GRAY), **style)
                elif world=='world3' and data['active']==0:
                    X = data['t']
                    #Xlabel('time')
                    xlim(0, TT)
                    Y = data['xh'][:, 1]
                    ylim(-0.5,0.6)
                    #ylabel('y of human')
                    plotit(X, Y, color= (BLACK if data['distracted']==0 else LIGHT_GRAY), **style)
            plotall()
    figure(fig1.number)
    tight_layout()
    savefig('plots/fig5-p1.pdf')
    figure(fig2.number)
    tight_layout()
    savefig('plots/fig5-p2.pdf')
    figure(fig3.number)
    savefig('plots/fig5-trajs.png', transparent=True)

def plot2():
    def setup():
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().xaxis.set_ticks_position('none')
        gca().yaxis.set_ticks_position('none')
        gca().spines['left'].set_linewidth(3.)
        gca().spines['bottom'].set_linewidth(3.)
        tick_params(axis='x', labelsize=20)
        tick_params(axis='y', labelsize=20)
    figure()
    setup()
    for data in datasets['world3']:
        if data['user']!=0:
            continue
        X = data['t']
        #xlabel('time', fontsize=30, fontweight='bold')
        xlim(0, 12)
        Y = data['xh'][:, 1]
        ylim(-0.5,0.6)
        #ylabel('y of human', fontsize=30, fontweight='bold')
        if data['active']==1:
            if data['distracted']==0:
                color = ORANGE_RED
            else:
                color = LIGHT_ORANGE
        else:
            if data['distracted']==0:
                color = BLACK
            else:
                color = LIGHT_GRAY
        if color in [BLACK, LIGHT_GRAY]:
            continue
        #plot(X, Y, color= color, linewidth=10., linestyle='dashed')
    tight_layout()
    savefig('plots/fig2.pdf')

#plot5()
#plot4()
#plot2()
#plot_world('world1')
#plot_world('world2')
#plot_world('world3')
#gen_csvs()

import matplotlib.font_manager as fm
palaprop = fm.FontProperties(fname='/Users/Nick/Downloads/Palatino-Roman.ttf')

def plotnick():
    fontsize=20
    def setup():
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

    gca().set_aspect(8)
    active = np.load("inten/intent-80-1486770219.pickle")
    #active = np.load("inten/intent-60-1486769041.pickle")
    passive = np.load("inten/intent-0-1486765163.pickle")

    setup()
    ylim(-0.1, 1.1)
    plot([active[2][1][t][1] for t in range(20)], color=ORANGE_RED, linewidth=2.)
    plot([passive[2][1][t][1] for t in range(20)], color=DARK_GRAY, linewidth=2.)
    ylabel('$b(\\theta=$merge$)$', fontproperties=palaprop, fontsize=fontsize)
    xlabel('time',fontproperties=palaprop, fontsize=fontsize)
    tight_layout()
    savefig('selfish_nick.pdf')


def plot_avg_y_delta():
    fontsize=20
    def setup():
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

    ydeltas = np.load("pertur/time_to_goals.npy")

    setup()
    plot([-.2, -.15, -.1, -0.05, 0,0.05, 0.1,0.15,0.2], ydeltas, marker='o', color=LIGHT_ORANGE)
    ylabel('Average vertical position H and R', fontproperties=palaprop, fontsize=fontsize)
    xlabel('relative starting position of human',fontproperties=palaprop, fontsize=fontsize)
    tight_layout()
    savefig('active_merge_varied.pdf')

#plotnick()
#plot_avg_y_delta()

def plot_nudging_lambda_beliefs():
    figure()
    fontsize=20
    def setup():
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
    #set_aspect(10)
    xlim(-25,225)
    setup()
    results = np.load("nudging_lambda_beliefs.npy")
    plot([0, 50, 100, 150, 200], results, marker="o", color=DARK_ORANGE, markersize=10,linewidth=2)
    ylabel('Final $b(\\theta=$attentive$)$',fontproperties=palaprop, fontsize=fontsize)
    xlabel("$\\lambda$",fontproperties=palaprop, fontsize=fontsize)
    tight_layout()
    savefig("nudging_lambda_beliefs")

plot_nudging_lambda_beliefs()

def equal_beliefs():
    figure()
    fontsize=20
    def setup():
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
    #gca().set_aspect(400)
    gca().set_aspect(100)
    xlim(-4,104)
    ylim(.6, 1.04)
    setup()
    results = np.load("dum/equal_beliefs.npy")
    plot([0, 10, 50, 100], [r[0] for i, r in enumerate(results[:5]) if i != 1], marker="o", color=BLACK, markersize=10,linewidth=2,markerfacecolor="white", markeredgewidth=2,)
    plot(0, results[0][0], color="gray")
    ylabel('$b(\\theta=$attentive$)$',fontproperties=palaprop, fontsize=fontsize)
    xticks([0, 10, 50, 100])
    #yticks([r[0] for r in results[:5]])
    xlabel("$\\lambda$",fontproperties=palaprop, fontsize=fontsize)
    tight_layout()
    savefig("equal_beliefs")

equal_beliefs()



def plot_cross_pos():
    figure()
    fontsize=20
    def setup():
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
    #set_aspect(10)
    #xlim(-25,225)
    setup()
    results = np.load("data/cross_final_x.npy")
    results = [-r for r in results]
    plot([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], results, marker="o", color=DARK_ORANGE, markersize=10,linewidth=2)
    ylabel('Final Robot $x$',fontproperties=palaprop, fontsize=fontsize)
    xlabel("Human Velocity",fontproperties=palaprop, fontsize=fontsize)
    title("Intersection")
    tight_layout()
    savefig("rss_cross_pos")

plot_cross_pos()

def plot_merge_pos():
    figure()
    fontsize=20
    def setup():
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
    #set_aspect(10)
    xlim(-.51,.31)
    ylim(0, -.12)
    setup()
    results = np.load("nudge/rss_merge_avg_deflections.npy")
    results = [-r for r in results]
    #fill_between((-.22, 22), (-.1, -.10), (-0.065, -0.065),  facecolor=LIGHT_GRAY)
    fill_between((-.52, 32), (-0.065, -0.065), (-0.13, -0.13),  facecolor=lighter(lighter(GREEN)))
    #plot((-.21, .21), (-.065, -.065), color=LIGHT_GRAY, linestyle='--')
    human_y = [-.2, -.15, -.1, -.05, 0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .60]
    delta_y = [hy - .3 for hy in human_y]
    plot(delta_y, results, marker="o", color=BLACK, markersize=10,linewidth=2,markerfacecolor="white", markeredgewidth=2)
    ylabel('Average $x$ of R',fontproperties=palaprop, fontsize=fontsize)
    xlabel("Initial $\\Delta y$ between R and H",fontproperties=palaprop, fontsize=fontsize)
    tight_layout()
    savefig("rss_merge_pos")

plot_merge_pos()

def plot_rewards():
    figure()
    fontsize=20
    def setup():
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
    #gca().set_aspect(400)
    #gca().set_aspect(100)
    xlim(-4,24)
    #ylim(.6, 1.04)
    setup()
    results = np.load("new_selfish_goal_rewards.npy")
    results = [results[i]+30 for i in range(len(results)) if i %2 == 0]
    plot([0, 4, 8, 12, 16, 20], results, marker="o", color=BLACK, markersize=10,linewidth=2,markerfacecolor="white", markeredgewidth=2)
    ylabel('$r_{goal}$',fontproperties=palaprop, fontsize=fontsize)
    xticks([0, 4, 8, 12, 16, 20])
    #yticks([r[0] for r in results[:5]])
    xlabel("$\\lambda$",fontproperties=palaprop, fontsize=fontsize)
    tight_layout()
    savefig("rewards")

plot_rewards()


def plot_cross_time():
    figure()
    fontsize=20
    def setup():
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
    #gca().set_aspect(400)
    #gca().set_aspect(100)
    xlim(-0.04,1.04)
    ylim(-.4, 7.4)
    setup()
    results = np.load("../driving-interactions/data/cross_time.npy")
    results = [-r for r in results]
    fill_between((-.06, 1.06), (-.44,-.44), (3,3),  facecolor=lighter(RED), color=RED)
    plot((-.21, .21), (3, 3), color=RED, linestyle='--')
    plot([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], results, marker="o", color=PURPLE, markersize=8, linewidth=2)
    ylabel('$\\Delta t$ between R and H Crossing',fontproperties=palaprop, fontsize=fontsize)
    #xticks([0, 4, 8, 12, 16, 20])
    #yticks([r[0] for r in results[:5]])
    xlabel("Intial $v$ of H",fontproperties=palaprop, fontsize=fontsize)
    tight_layout()
    savefig("rss_cross_time")

plot_cross_time()

def plot_cross_dist():
    figure()
    fontsize=20
    def setup():
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')
    #gca().set_aspect(400)
    #gca().set_aspect(100)
    xlim(-0.04,1.04)
    ylim(.0, .4)
    setup()
    results = np.load("../driving-interactions/data/cross_dist.npy")
    fill_between((-.06, 1.06), (.13,.13), (0,0),  facecolor=lighter(lighter(RED)), color=RED)
    plot([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], results, marker="o", color=BLACK, markersize=10, markerfacecolor="white", markeredgewidth=2, linewidth=2)
    ylabel('Minimum distance between R and H',fontproperties=palaprop, fontsize=fontsize)
    #xticks([0, 4, 8, 12, 16, 20])
    #yticks([r[0] for r in results[:5]])
    xlabel("Intial $v$ of H",fontproperties=palaprop, fontsize=fontsize)
    tight_layout()
    savefig("rss_cross_dist")

plot_cross_dist()
