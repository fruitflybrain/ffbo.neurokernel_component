#!/usr/bin/env python

import neurokernel.mpi_relaunch
import sys, traceback
import os
import pickle
import math
import time
import six
import simplejson as json
import ast
from pathlib import Path
from time import gmtime, strftime
from configparser import ConfigParser
import importlib
import inspect
import argparse
import subprocess
import urllib
import requests
import logging
import threading

import txaio
import OpenSSL.crypto
from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.internet._sslverify import OpenSSLCertificateAuthorities
from twisted.internet.ssl import CertificateOptions
from twisted.logger import Logger
import autobahn
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from autobahn.wamp.exception import ApplicationError
from autobahn.wamp import auth
from autobahn.websocket.protocol import WebSocketProtocol
from autobahn.wamp.types import RegisterOptions

from autobahn_sync import AutobahnSync

import networkx as nx
import numpy as np
import pycuda.driver as cuda
import h5py

from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core
from neurokernel.pattern import Pattern
from neurokernel.tools.timing import Timer
from neurokernel.LPU.LPU import LPU
from neurokernel.tools.misc import LPUExecutionError
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
from neuroarch import nk
from version import __version__


import msgpack
import msgpack_numpy
msgpack_numpy.patch()

root = os.path.expanduser("/")
home = os.path.expanduser("~")
filepath = os.path.dirname(os.path.abspath(__file__))
config_files = []
config_files.append(os.path.join(home, ".ffbo/config", "ffbo.neuroarch_component.ini"))
config_files.append(os.path.join(root, ".ffbo/config", "ffbo.neuroarch_component.ini"))
config_files.append(os.path.join(home, ".ffbo/config", "config.ini"))
config_files.append(os.path.join(root, ".ffbo/config", "config.ini"))
config_files.append(os.path.join(filepath, "..", "config.ini"))
config = ConfigParser()
configured = False
file_type = 0
for config_file in config_files:
    if os.path.exists(config_file):
        config.read(config_file)
        configured = True
        break
    file_type += 1
if not configured:
    raise Exception("No config file exists for this component")

user = config["USER"]["user"]
secret = config["USER"]["secret"]
ssl = eval(config["AUTH"]["ssl"])
websockets = "wss" if ssl else "ws"
if "ip" in config["SERVER"]:
    ip = config["SERVER"]["ip"]
else:
    ip = "localhost"
if ip in ["localhost", "127.0.0.1"]:
    port = config["NLP"]["port"]
else:
    port = config["NLP"]["expose-port"]
url =  "{}://{}:{}/ws".format(websockets, ip, port)
realm = config["SERVER"]["realm"]
authentication = eval(config["AUTH"]["authentication"])
debug = eval(config["DEBUG"]["debug"])
ca_cert_file = config["AUTH"]["ca_cert_file"]
#intermediate_cert_file = config["AUTH"]["intermediate_cert_file"]

#logging.basicConfig()
#logging.getLogger("twisted").setLevel(logging.CRITICAL)

def create_graph_from_database_returned(x):
    """Builds a NetworkX graph using processed data from NeuroArch.

    # Arguments:
        x (dict): File url.

    # Returns:
        g (NetworkX MultiDiGraph): A MultiDiGraph instance with the circuit graph.
    """
    g = nx.MultiDiGraph()
    g.add_nodes_from(x["nodes"].items())
    for pre, v, attrs in x["edges"]:
        g.add_edge(pre, v, **attrs)
    return g


class neurokernel_server(object):
    """ A Neurokernel server launcher instance. """

    def __init__(self):
        cuda.init()
        self.ngpus = cuda.Device.count()
        if self.ngpus <= 0:
            raise ValueError("No GPU found on this device")

    def launch(self, user_id, task):
        # neuron_uid_list = [str(a) for a in task['neuron_list']]
        try:
            # conf_obj = get_config_obj()
            # config = conf_obj.conf

            setup_logger(file_name = 'neurokernel_'+user_id+'.log', screen = False)

            manager = core.Manager()

            lpus = {}
            patterns = {}
            G = task['data']

            for i in list(G['Pattern'].keys()):
                a = G['Pattern'][i]['nodes']
                if len([k for k,v in a.items() if v['class'] == 'Port']) == 0:
                    del G['Pattern'][i]

            for i in list(G['LPU'].keys()):
                a = G['LPU'][i]['nodes']
                if len(a) < 3:
                    del G['LPU'][i]

            # with open('G.pickle', 'wb') as f:
            #     pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
            # print(G)
            # print(G.keys())
            # print(G['LPU'])
            # print(G['LPU'].keys())

            # get graph and output_uid_list for each LPU
            for k, lpu in G['LPU'].items():
                lpus[k] = {}
                g_lpu_na = create_graph_from_database_returned(lpu)
                lpu_nk_graph = nk.na_lpu_to_nk_new(g_lpu_na)
                lpus[k]['graph'] = lpu_nk_graph
                # lpus[k]['output_uid_list'] = list(
                #             set(lpu_nk_graph.nodes()).intersection(
                #                 set(neuron_uid_list)))
                # lpus[k]['output_file'] = '{}_output_{}.h5'.format(k, user_id)

            for kkey, lpu in lpus.items():
                graph = lpu['graph']

                for uid, comp in graph.nodes.items():
                    if 'attr_dict' in comp:
                        print('Found attr_dict; fixing...')
                        nx.set_node_attributes(graph, {uid: comp['attr_dict']})
                        # print('changed',uid)
                        graph.nodes[uid].pop('attr_dict')
                    if 'params' in comp:
                        params = graph.nodes[uid].pop('params')
                        nx.set_node_attributes(graph, {uid: {k: float(v) for k, v in params.items()}})
                    if 'states' in comp:
                        states = graph.nodes[uid].pop('states')
                        nx.set_node_attributes(graph, {uid: {'init{}'.format(k): float(v) for k, v in states.items()}})
                for i,j,k,v in graph.edges(keys=True, data=True):
                    if 'attr_dict' in v:
                        for key in v['attr_dict']:
                            nx.set_edge_attributes(graph, {(i,j,k): {key: v['attr_dict'][key]}})
                        graph.edges[(i,j,k)].pop('attr_dict')
                lpus[kkey]['graph'] = graph

            # get graph for each Pattern
            for k, pat in G['Pattern'].items():
                l1,l2 = k.split('-')
                if l1 in lpus and l2 in lpus:
                    g_pattern_na = create_graph_from_database_returned(pat)
                    pattern_nk = nk.na_pat_to_nk(g_pattern_na)
                    #print(lpus[l1]['graph'].nodes(data=True))
                    lpu_ports = [node[1]['selector'] \
                                 for node in lpus[l1]['graph'].nodes(data=True) \
                                 if node[1]['class']=='Port'] + \
                                [node[1]['selector'] \
                                 for node in lpus[l2]['graph'].nodes(data=True) \
                                 if node[1]['class']=='Port']
                    pattern_ports = pattern_nk.nodes()
                    patterns[k] = {}
                    patterns[k]['graph'] = pattern_nk.subgraph(
                        list(set(lpu_ports).intersection(set(pattern_ports))))

            # dt = config['General']['dt']
            # if 'dt' in task:
            dt = task['dt']
            print(dt)

            device_count = 0
            # add LPUs to manager
            for k, lpu in lpus.items():
                lpu_name = k
                graph = lpu['graph']

                for uid, comp in graph.nodes.items():
                    if 'attr_dict' in comp:
                        nx.set_node_attributes(graph, {uid: comp['attr_dict']})
                        # print('changed',uid)
                        graph.nodes[uid].pop('attr_dict')
                for i,j,ko,v in graph.edges(keys=True, data=True):
                    if 'attr_dict' in v:
                        for key in v['attr_dict']:
                            nx.set_edge_attributes(graph, {(i,j,ko): {key: v['attr_dict'][key]}})
                        graph.edges[(i,j,ko)].pop('attr_dict')
                # nx.write_gexf(graph,'name.gexf')
                # with open(lpu_name + '.pickle', 'wb') as f:
                #     pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
                comps =  graph.nodes.items()

                #for uid, comp in comps:
                #    if 'attr_dict' in comp:
                #        nx.set_node_attributes(graph, {uid: comp['attr_dict']})
                #        print('changed',uid)
                #    if 'class' in comp:

                # if k == 'retina':
                #     if config['Retina']['intype'] == 'Natural':
                #         coord_file = config['InputType']['Natural']['coord_file']
                #         tmp = os.path.splitext(coord_file)
                #         config['InputType']['Natural']['coord_file'] = '{}_{}{}'.format(
                #                 tmp[0], user_id, tmp[1])
                #     prs = [node for node in graph.nodes(data=True) \
                #            if node[1]['class'] == 'PhotoreceptorModel']
                #     for pr in prs:
                #         graph.node[pr[0]]['num_microvilli'] = 3000
                #     input_processors = [RetinaInputIndividual(config, prs, user_id)]
                #     extra_comps = [PhotoreceptorModel]
                #     retina_input_uids = [a[0] for a in prs]
                # # elif k == 'EB':
                # #     input_processor = StepInputProcessor('I', [node[0] for node in graph.nodes(data=True) \
                # #            if node[1]['class'] == 'LeakyIAF'], 40.0, 0.0, 1.0)
                # #     input_processors = [input_processor]
                # #     extra_comps = []#[BufferVoltage]
                # else:
                #     input_processors = []
                #     extra_comps = [BufferVoltage]
                if 'inputProcessors' in task:
                    if lpu_name in task['inputProcessors']:
                        input_processors, record = \
                            loadInputProcessors(task['inputProcessors'][lpu_name])
                        lpus[k]['input_record'] = record
                    else:
                        input_processors = []
                else:
                    input_processors = []

                # configure output processors
                lpus[k]['output_file'] = '{}_output_{}.h5'.format(k, user_id)
                output_processors = []
                if 'outputProcessors' in task:
                    if lpu_name in task['outputProcessors']:
                        output_processors, record = loadOutputProcessors(
                                                lpus[k]['output_file'],
                                                task['outputProcessors'][lpu_name])
                        if len(record):
                            lpus[k]['output_uid_dict'] = record

                # (comp_dict, conns) = LPU.graph_to_dicts(graph)
                manager.add(LPU, k, dt, 'pickle', pickle.dumps(graph),#comp_dict, conns,
                            device = device_count, input_processors = input_processors,
                            output_processors = output_processors,
                            extra_comps = [], debug = False)
                device_count = (device_count+1) % self.ngpus

            # connect LPUs by Patterns
            for k, pattern in patterns.items():
                l1,l2 = k.split('-')
                if l1 in lpus and l2 in lpus:
                    print('Connecting {} and {}'.format(l1, l2))
                    pat, key_order = Pattern.from_graph(nx.DiGraph(pattern['graph']),
                                                        return_key_order = True)
                    print(l1,l2)
                    print(key_order)
                    with Timer('update of connections in Manager'):
                        try:
                            manager.connect(l1, l2, pat,
                                            int_0 = key_order.index('{}/{}'.format(k,l1)),
                                            int_1 = key_order.index('{}/{}'.format(k,l2)))
                        except ValueError:
                            manager.connect(l1, l2, pat,
                                            int_0 = key_order.index(l1),
                                            int_1 = key_order.index(l2))

            # start simulation
            # steps = config['General']['steps']
            # ignored_steps = config['General']['ignored_steps']
            # if 'steps' in task:
            steps = task['steps']
            # if 'ignored_steps' in task:
            # ignored_steps = task['ignored_steps']
            # ignored_steps = 0
            # steps = 100
            manager.spawn()
            manager.start(steps=steps)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            print('An error occured during the compilation\n' + tb)
            return {'error':
                        {'exception': tb,
                         'message': 'An error occured during the compilation in execution'}}

        try:
            manager.wait()
        except LPUExecutionError:
            ErrorMessage = 'An error occured during execution of LPU {} at step {}:\n'.format(
                            manager._errors[0][0], manager._errors[0][1]) + \
                            ''.join(manager._errors[0][2])
            print(ErrorMessage)
            return {'error':
                        {'exception': ''.join(manager._errors[0][2]),
                         'message': 'An error occured during execution of LPU {} at step {}:\n'.format(
                            manager._errors[0][0], manager._errors[0][1])}}
        time.sleep(5)
        # print(task)

        ignored_steps = 0

        try:
            # post-processing inputs (hard coded, can be better organized)
            result = {'sensory': {}, 'input': {}, 'output': {}}
            for k, lpu in lpus.items():
                records = lpu.get('input_record', [])
                for record in records:
                    if record['sensory_file'] is not None:
                        if k not in result['sensory']:
                            result['sensory'][k] = []
                        with h5py.File(record['sensory_file']) as sensory_file:
                            result['sensory'][k].append({'dt': record['sensory_interval']*dt,
                                                         'data': sensory_file['sensory'][:]})
                    if record['input_file'] is not None:
                        with h5py.File(record['input_file']) as input_file:
                            sample_interval = input_file['metadata'].attrs['sample_interval']
                            for var in input_file.keys():
                                if var == 'metadata': continue
                                uids = [n.decode() for n in input_file[var]['uids'][:]]
                                input_array = input_file[var]['data'][:]
                                for i, item in enumerate(uids):
                                    if var == 'spike_state':
                                        input = np.nonzero(input_array[ignored_steps:, i:i+1].reshape(-1))[0]*dt
                                        if item in result['input']:
                                            if 'spike_time' in result['input'][item]:
                                                result['input'][item]['spike_time']['data'].append(input)
                                                result['input'][item]['spike_time']['data'] = \
                                                    np.sort(result['input'][item]['spike_time']['data'])
                                            else:
                                                result['input'][item]['spike_time'] = {
                                                    'data': input.copy(),
                                                    'dt': dt*sample_interval}
                                        else:
                                            result['input'][item] = {'spike_time': {
                                                'data': input.copy(),
                                                'dt': dt*sample_interval}}
                                    else:
                                        input = input_array[ignored_steps:, i:i+1]
                                        if item in result['input']:
                                            if var in result['input'][item]:
                                                result['input'][item][var]['data'] += input
                                            else:
                                                result['input'][item][var] = {
                                                    'data': input.copy(),
                                                    'dt': dt*sample_interval}
                                        else:
                                            result['input'][item] = {var: {
                                                'data': input.copy(),
                                                'dt': dt*sample_interval}}

            # if 'retina' in lpus:
            #     input_array = si.read_array(
            #             '{}_{}.h5'.format(config['Retina']['input_file'], user_id))
            #     inputs[u'ydomain'] = input_array.max()
            #     for i, item in enumerate(retina_input_uids):
            #         inputs['data'][item] = np.hstack(
            #             (np.arange(int((steps-ignored_steps)/10)).reshape((-1,1))*dt*10,
            #              input_array[ignored_steps::10,i:i+1])).tolist()
            #
            #     del input_array

            # post-processing outputs from all LPUs and combine them into one dictionary
            # result = {u'data': {}}

            for k, lpu in lpus.items():
                uid_dict = lpu.get('output_uid_dict', None)
                if uid_dict is not None:
                    with h5py.File(lpu['output_file']) as output_file:
                        sample_interval = output_file['metadata'].attrs['sample_interval']
                        for var in uid_dict:
                            if var == 'spike_state':
                                uids = [n.decode() for n in output_file[var]['uids'][:]]
                                spike_times = output_file[var]['data']['time'][:]
                                index = output_file[var]['data']['index'][:]
                                for i, item in enumerate(uids):
                                    # output = np.nonzero(output_array[ignored_steps:, i:i+1].reshape(-1))[0]*dt
                                    output = spike_times[index == i]
                                    output = output[output>ignored_steps*dt]-ignored_steps*dt
                                    if item in result['output']:
                                        result['output'][item]['spike_time'] = {
                                            'data': output,
                                            'dt': dt*sample_interval}
                                    else:
                                        result['output'][item] = {'spike_time': {
                                                    'data': output,
                                                    'dt': dt*sample_interval}}
                            else:
                                uids = [n.decode() for n in output_file[var]['uids'][:]]
                                output_array = output_file[var]['data'][:]
                                for i, item in enumerate(uids):
                                    # if var == 'spike_state':
                                    #     output = np.nonzero(output_array[ignored_steps:, i:i+1].reshape(-1))[0]*dt
                                    #     if item in result['output']:
                                    #         result['output'][item]['spike_time'] = {
                                    #             'data': output.tolist(),
                                    #             'dt': dt}
                                    #     else:
                                    #         result['output'][item] = {'spike_time': {
                                    #             'data': output.tolist(),
                                    #             'dt': dt}}
                                    # else:
                                    output = output_array[ignored_steps:, i:i+1]
                                    if item in result['output']:
                                        result['output'][item][var] = {
                                            'data': output.copy(),
                                            'dt': dt*sample_interval}
                                    else:
                                        result['output'][item] = {var: {
                                            'data': output.copy(),
                                            'dt': dt*sample_interval}}
            result = {'success': {'result': result, 'meta': {'dur': steps*dt}}}
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            print('An error occured postprocessing of results\n' + tb)
            return {'error':
                        {'exception': tb,
                         'message': 'An error occured when postprocessing of results in execution'}}
        return result

def loadInputProcessors(X):
    """
    Load Input Processors for 1 LPU.

    Parameters
    ----------
    X: List of dictionaries
       Each dictionary contains the following key/value pairs:
       'module': str,
                 specifying the module that the InputProcessor class
                 can be imported
       'class': str,
                name of the InputProcessor class.
       and other keys should correspond to the arguments of the InputProcessor
    """
    inList = []
    record = []
    for a in X:
        d = importlib.import_module(a.pop('module'))
        processor = getattr(d, a.pop('class'))
        sig = inspect.signature(processor)
        arg_dict = {param_name: a.get(param_name) if param.default is param.empty\
                    else a.get(param_name, param.default) \
                    for param_name, param in sig.parameters.items()}
        input_processor = processor(**arg_dict)
        inList.append(input_processor)
        record.append(input_processor.record_settings)
    # for a in X:
    #     if a['name'] == 'InIGaussianNoise':
    #         inList.append(InIGaussianNoise(a['node_id'], a['mean'], a['std'], a['t_start'], a['t_end']))
    #     elif a['name'] == 'InISinusoidal':
    #         inList.append(InISinusoidal(a['node_id'], a['amplitude'], a['frequency'], a['t_start'], a['t_end'], a['mean'], a['shift'], a['frequency_sweep'], a['frequency_sweep_frequency'], a['threshold_active'], a['threshold_value']))
    #     elif a['name'] == 'InIBoxcar':
    #         inList.append(InIBoxcar(a['node_id'], a['I_val'], a['t_start'], a['t_end']))
    #     elif a['name'] == 'StepInputProcessor':
    #         inList.append(StepInputProcessor(a['variable'], a['uids'], a['val'], a['start'], a['stop']))
    #     elif a['name'] == 'BU_InputProcessor':
    #         inList.append(BU_InputProcessor(a['shape'], a['dt'], a['dur'], a['id'], a['video_config'],
    #                                         a['rf_config'], a['neurons']))
    #     elif a['name'] == 'PB_InputProcessor':
    #         inList.append(PB_InputProcessor(a['shape'], a['dt'], a['dur'], a['id'], a['video_config'],
    #                                         a['rf_config'], a['neurons']))
    return inList, record


def loadOutputProcessors(filename, outputProcessor_dicts):
    outList = []
    record = {}
    for a in outputProcessor_dicts:
        outprocessor_class = a.get('class')
        if outprocessor_class == 'Record':
            to_record = [(k, v['uids']) for k, v in a['uid_dict'].items()]
            processor = FileOutputProcessor(to_record,
                                            filename,
                                            sample_interval = a.get('sample_interval', 1))
            outList.append(processor)
            record = a['uid_dict']
        else:
            d = importlib.import_module(a.get('module'))
            processor = getattr(d, outprocessor_class)
            sig = inspect.signature(processor)
            arg_dict = {param_name: a.get(param_name) if param.default is param.empty\
                        else a.get(param_name, param.default) \
                        for param_name, param in sig.parameters.items()}
            outList.append(processor(**arg_dict))
    return outList, record


class AppSession(ApplicationSession):

    log = Logger()

    def onConnect(self):
        setProtocolOptions(self._transport,
                           maxFramePayloadSize = 0,
                           maxMessagePayloadSize = 0,
                           autoFragmentSize = 65536)
        if self.config.extra['auth']:
            self.join(self.config.realm, [u"wampcra"], user)
        else:
            self.join(self.config.realm, [], user)

    def onChallenge(self, challenge):
        if challenge.method == u"wampcra":
            #print("WAMP-CRA challenge received: {}".format(challenge))

            if u'salt' in challenge.extra:
                # salted secret
                key = auth.derive_key(secret,
                                      challenge.extra['salt'],
                                      challenge.extra['iterations'],
                                      challenge.extra['keylen'])
            else:
                # plain, unsalted secret
                key = secret

            # compute signature for challenge, using the key
            signature = auth.compute_wcs(key, challenge.extra['challenge'])

            # return the signature to the router for verification
            return signature

        else:
            raise Exception("Invalid authmethod {}".format(challenge.method))

    @inlineCallbacks
    def onJoin(self, details):
        server = neurokernel_server()
        launch_queue = []

        def nk_receive_task(task, details=None):
            user_id = str(task["user"])
            launch_queue.append((user_id, task))
            return({'success': 'Job received. Currently queued #{}'.format(len(launch_queue))})

        uri = six.u('ffbo.nk.launch.%s' % str(details.session))
        yield self.register(nk_receive_task, uri)
        self.log.info('procedure %s registered' % uri)

        # Listen for ffbo.processor.connected
        @inlineCallbacks
        def register_component():
            self.log.info( "Registering neurokernel component")
            # CALL server registration
            try:
                # registered the procedure we would like to call
                res = yield self.call(six.u('ffbo.server.register'),
                                      details.session,
                                      'nk',
                                      {"name": 'neurokernel',
                                       "version": __version__,
                                       "autobahn": autobahn.__version__})
                self.log.info("register new server called with result: {result}", result=res)

            except ApplicationError as e:
                if e.error != 'wamp.error.no_such_procedure':
                    raise e

        def process_queue():
            if len(launch_queue):
                user_id, task = launch_queue[0]
                launch_queue.pop(0)
                res = server.launch(user_id, task)
                batch_size = 1024*1024*100
                try:
                    res_to_processor = self.call(six.u(task['forward']), msgpack.packb({'execution_result_start': six.u(task['name'])}))
                    packed = msgpack.packb(res)
                    for i in range(0, len(packed), batch_size):
                        res_processor = self.call(six.u(task['forward']), packed[i:i+batch_size])
                    res_to_processor = self.call(six.u(task['forward']), msgpack.packb({'execution_result_end':six.u(task['name'])}))
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                    print('An error occured sending results back to FBL Client\n' + tb)
                    errmsg = {'error':
                                {'exception': tb,
                                 'message': 'An error occured when sending results back to FBL Client'}}
                    try:
                        res_to_processor = self.call(six.u(task['forward']), msgpack.packb({'execution_result_start': six.u(task['name'])}))
                        packed = msgpack.packb(errmsg)
                        for i in range(0, len(packed), batch_size):
                            res_processor = self.call(six.u(task['forward']), packed[i:i+batch_size])
                        res_to_processor = self.call(six.u(task['forward']), msgpack.packb({'execution_result_end':six.u(task['name'])}))
                    except:
                        pass
            else:
                return

        def persist_processing():
            while(True):
                process_queue()
                time.sleep(1)

        register_component()
        x = threading.Thread(target = persist_processing, args = ())
        x.start()


def setProtocolOptions(transport,
                       version=None,
                       utf8validateIncoming=None,
                       acceptMaskedServerFrames=None,
                       maskClientFrames=None,
                       applyMask=None,
                       maxFramePayloadSize=None,
                       maxMessagePayloadSize=None,
                       autoFragmentSize=None,
                       failByDrop=None,
                       echoCloseCodeReason=None,
                       serverConnectionDropTimeout=None,
                       openHandshakeTimeout=None,
                       closeHandshakeTimeout=None,
                       tcpNoDelay=None,
                       perMessageCompressionOffers=None,
                       perMessageCompressionAccept=None,
                       autoPingInterval=None,
                       autoPingTimeout=None,
                       autoPingSize=None):
    """ from autobahn.websocket.protocol.WebSocketClientFactory.setProtocolOptions """
    transport.factory.setProtocolOptions(
            version = version,
            utf8validateIncoming = utf8validateIncoming,
            acceptMaskedServerFrames = acceptMaskedServerFrames,
            maskClientFrames = maskClientFrames,
            applyMask = applyMask,
            maxFramePayloadSize = maxFramePayloadSize,
            maxMessagePayloadSize = maxMessagePayloadSize,
            autoFragmentSize = autoFragmentSize,
            failByDrop = failByDrop,
            echoCloseCodeReason = echoCloseCodeReason,
            serverConnectionDropTimeout = serverConnectionDropTimeout,
            openHandshakeTimeout = openHandshakeTimeout,
            closeHandshakeTimeout = closeHandshakeTimeout,
            tcpNoDelay = tcpNoDelay,
            perMessageCompressionOffers = perMessageCompressionOffers,
            perMessageCompressionAccept = perMessageCompressionAccept,
            autoPingInterval = autoPingInterval,
            autoPingTimeout = autoPingTimeout,
            autoPingSize = autoPingSize)

    if version is not None:
        if version not in WebSocketProtocol.SUPPORTED_SPEC_VERSIONS:
            raise Exception("invalid WebSocket draft version %s (allowed values: %s)" % (version, str(WebSocketProtocol.SUPPORTED_SPEC_VERSIONS)))
        if version != transport.version:
            transport.version = version

    if utf8validateIncoming is not None and utf8validateIncoming != transport.utf8validateIncoming:
        transport.utf8validateIncoming = utf8validateIncoming

    if acceptMaskedServerFrames is not None and acceptMaskedServerFrames != transport.acceptMaskedServerFrames:
        transport.acceptMaskedServerFrames = acceptMaskedServerFrames

    if maskClientFrames is not None and maskClientFrames != transport.maskClientFrames:
        transport.maskClientFrames = maskClientFrames

    if applyMask is not None and applyMask != transport.applyMask:
        transport.applyMask = applyMask

    if maxFramePayloadSize is not None and maxFramePayloadSize != transport.maxFramePayloadSize:
        transport.maxFramePayloadSize = maxFramePayloadSize

    if maxMessagePayloadSize is not None and maxMessagePayloadSize != transport.maxMessagePayloadSize:
        transport.maxMessagePayloadSize = maxMessagePayloadSize

    if autoFragmentSize is not None and autoFragmentSize != transport.autoFragmentSize:
        transport.autoFragmentSize = autoFragmentSize

    if failByDrop is not None and failByDrop != transport.failByDrop:
        transport.failByDrop = failByDrop

    if echoCloseCodeReason is not None and echoCloseCodeReason != transport.echoCloseCodeReason:
        transport.echoCloseCodeReason = echoCloseCodeReason

    if serverConnectionDropTimeout is not None and serverConnectionDropTimeout != transport.serverConnectionDropTimeout:
        transport.serverConnectionDropTimeout = serverConnectionDropTimeout

    if openHandshakeTimeout is not None and openHandshakeTimeout != transport.openHandshakeTimeout:
        transport.openHandshakeTimeout = openHandshakeTimeout

    if closeHandshakeTimeout is not None and closeHandshakeTimeout != transport.closeHandshakeTimeout:
        transport.closeHandshakeTimeout = closeHandshakeTimeout

    if tcpNoDelay is not None and tcpNoDelay != transport.tcpNoDelay:
        transport.tcpNoDelay = tcpNoDelay

    if perMessageCompressionOffers is not None and pickle.dumps(perMessageCompressionOffers) != pickle.dumps(transport.perMessageCompressionOffers):
        if type(perMessageCompressionOffers) == list:
            #
            # FIXME: more rigorous verification of passed argument
            #
            transport.perMessageCompressionOffers = copy.deepcopy(perMessageCompressionOffers)
        else:
            raise Exception("invalid type %s for perMessageCompressionOffers - expected list" % type(perMessageCompressionOffers))

    if perMessageCompressionAccept is not None and perMessageCompressionAccept != transport.perMessageCompressionAccept:
        transport.perMessageCompressionAccept = perMessageCompressionAccept

    if autoPingInterval is not None and autoPingInterval != transport.autoPingInterval:
        transport.autoPingInterval = autoPingInterval

    if autoPingTimeout is not None and autoPingTimeout != transport.autoPingTimeout:
        transport.autoPingTimeout = autoPingTimeout

    if autoPingSize is not None and autoPingSize != transport.autoPingSize:
        assert(type(autoPingSize) == float or type(autoPingSize) == int)
        assert(4 <= autoPingSize <= 125)
        transport.autoPingSize = autoPingSize



if __name__ == '__main__':
    from twisted.internet._sslverify import OpenSSLCertificateAuthorities
    from twisted.internet.ssl import CertificateOptions
    import OpenSSL.crypto
    import getpass

    # parse command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output.')
    parser.add_argument('--url', dest='url', type=six.text_type, default=url,
                        help='The router URL (defaults to value from config.ini)')
    parser.add_argument('--realm', dest='realm', type=six.text_type, default=realm,
                        help='The realm to join (defaults to value from config.ini).')
    parser.add_argument('--ca_cert', dest='ca_cert_file', type=six.text_type,
                        default=ca_cert_file,
                        help='Root CA PEM certificate file (defaults to value from config.ini).')
    # parser.add_argument('--int_cert', dest='intermediate_cert_file', type=six.text_type,
    #                     default=intermediate_cert_file,
    #                     help='Intermediate PEM certificate file (defaults to value from config.ini).')
    parser.add_argument('--no-ssl', dest='ssl', action='store_false')
    parser.set_defaults(ssl=ssl)
    parser.set_defaults(debug=debug)

    args = parser.parse_args()

    # start logging
    if args.debug:
        txaio.start_logging(level='debug')
    else:
        txaio.start_logging(level='info')

    # any extra info we want to forward to our ClientSession (in self.config.extra)
    extra = {'auth': True}

    if args.ssl:
        st_cert=open(args.ca_cert_file, 'rt').read()
        c=OpenSSL.crypto
        ca_cert=c.load_certificate(c.FILETYPE_PEM, st_cert)

        # st_cert=open(args.intermediate_cert_file, 'rt').read()
        # intermediate_cert=c.load_certificate(c.FILETYPE_PEM, st_cert)

        certs = OpenSSLCertificateAuthorities([ca_cert]) #, intermediate_cert])
        ssl_con = CertificateOptions(trustRoot=certs)

        # now actually run a WAMP client using our session class ClientSession
        runner = ApplicationRunner(url=args.url, realm=args.realm, extra=extra, ssl=ssl_con)

    else:
        # now actually run a WAMP client using our session class ClientSession
        runner = ApplicationRunner(url=args.url, realm=args.realm, extra=extra)

    runner.run(AppSession, auto_reconnect=True)
