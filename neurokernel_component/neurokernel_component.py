#!/usr/bin/env python

import sys

from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.logger import Logger

from autobahn.twisted.util import sleep
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from autobahn.wamp.exception import ApplicationError
from autobahn.wamp import auth

import subprocess
import os
import argparse
import six
import txaio
import time
import simplejson as json
import pickle
import traceback

# Neurokernel Imports
import numpy as np
import h5py
import networkx as nx
import pycuda.driver as cuda

from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core
from neurokernel.pattern import Pattern
from neurokernel.tools.timing import Timer
from neurokernel.LPU.LPU import LPU
from retina.InputProcessors.RetinaInputIndividual import RetinaInputIndividual
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
import neurokernel.LPU.utils.simpleio as si
from neuroarch import nk

from retina.configreader import ConfigReader
from retina.NDComponents.MembraneModels.PhotoreceptorModel import PhotoreceptorModel
from retina.NDComponents.MembraneModels.BufferPhoton import BufferPhoton
from retina.NDComponents.MembraneModels.BufferVoltage import BufferVoltage

from config import *

def get_config_obj():
    conf_name = 'configuration/retina.cfg'

    # append file extension if not exist
    conf_filename = conf_name if '.' in conf_name else ''.join(
        [conf_name, '.cfg'])
    conf_specname = 'configuration/retina_template.cfg'

    return ConfigReader(conf_filename, conf_specname)


def create_graph_from_database_returned(dict):
    g = nx.MultiDiGraph()
    g.add_nodes_from(dict['nodes'].items())
    for pre,v in dict['edges'].items():
        if len(v) > 0:
            for post, edges in v.items():
                for edge_attr in edges.itervalues():
                    g.add_edge(pre, post, attr_dict = edge_attr)
    return g


class neurokernel_server(object):
    """ Various convenience methods to launch neurokernel examples """

    def __init__(self):
        cuda.init()
        if cuda.Device.count() < 0:
            raise ValueError("No GPU found on this device")

    def launch(self, user_id, task):
        neuron_uid_list = [str(a) for a in task['neuron_list']]
        
        conf_obj = get_config_obj()
        config = conf_obj.conf
        
        if config['Retina']['intype'] == 'Natural':
            coord_file = config['InputType']['Natural']['coord_file']
            tmp = os.path.splitext(coord_file)
            config['InputType']['Natural']['coord_file'] = '{}_{}{}'.format(
                    tmp[0], user_id, tmp[1])
        
        setup_logger(file_name = 'neurokernel_'+user_id+'.log', screen = False)
        
        manager = core.Manager()
        
        lpus = {}
        patterns = {}
        G = task
        
        # get graph and output_uid_list for each LPU
        for k, lpu in G['LPU'].iteritems():
            lpus[k] = {}
            g_lpu_na = create_graph_from_database_returned(lpu)
            lpu_nk_graph = nk.na_lpu_to_nk_new(g_lpu_na)
            lpus[k]['graph'] = lpu_nk_graph
            lpus[k]['output_uid_list'] = list(
                        set(lpu_nk_graph.nodes()).intersection(
                            set(neuron_uid_list)))
            lpus[k]['output_file'] = '{}_output_{}.h5'.format(k, user_id)
        
        # get graph for each Pattern
        for k, pat in G['Pattern'].iteritems():
            l1,l2 = k.split('-')
            if l1 in lpus and l2 in lpus:
                g_pattern_na = create_graph_from_database_returned(pat)
                pattern_nk = nk.na_pat_to_nk(g_pattern_na)
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
        
        dt = config['General']['dt']

        # add LPUs to manager
        for k, lpu in lpus.iteritems():
            graph = lpu['graph']
            if k == 'retina':
                prs = [node for node in graph.nodes(data=True) \
                       if node[1]['class'] == 'PhotoreceptorModel']
                for pr in prs:
                    graph.node[pr[0]]['num_microvilli'] = 3000
                input_processors = [RetinaInputIndividual(config, prs, user_id)]
                extra_comps = [PhotoreceptorModel]
                retina_input_uids = [a[0] for a in prs]
            else:
                input_processors = []
                extra_comps = [BufferVoltage]
            output_processor = FileOutputProcessor(
                                    [('V', lpu['output_uid_list'])],
                                    lpu['output_file'], sample_interval=10)
        
            (comp_dict, conns) = LPU.graph_to_dicts(graph)
            
            manager.add(LPU, k, dt, comp_dict, conns,
                        device = 0, input_processors = input_processors,
                        output_processors = [output_processor],
                        extra_comps = extra_comps)

        # connect LPUs by Patterns
        for k, pattern in patterns.iteritems():
            l1,l2 = k.split('-')
            if l1 in lpus and l2 in lpus:
                print('Connecting {} and {}'.format(l1, l2))
                pat, key_order = Pattern.from_graph(nx.DiGraph(pattern['graph']))
                with Timer('update of connections in Manager'):
                    manager.connect(l1, l2, pat,
                                    int_0 = key_order.index(l1),
                                    int_1 = key_order.index(l2))
        
        # start simulation
        steps = config['General']['steps']
        ignored_steps = config['General']['ignored_steps']
        manager.spawn()
        manager.start(steps=steps)
        manager.wait()

        time.sleep(5)
        
        # post-processing inputs (hard coded, can be better organized)
        inputs = {u'ydomain': 1.0,
                      u'xdomain': dt*(steps-ignored_steps),
                      u'dt': dt*10,
                      u'data': {}}
        if 'retina' in lpus:
            input_array = si.read_array(
                    '{}_{}.h5'.format(config['Retina']['input_file'], user_id))
            inputs[u'ydomain'] = input_array.max()
            for i, item in enumerate(retina_input_uids):
                inputs['data'][item] = np.hstack(
                    (np.arange(int((steps-ignored_steps)/10)).reshape((-1,1))*dt*10,
                     input_array[ignored_steps::10,i:i+1])).tolist()
            
            del input_array
        
        # post-processing outputs from all LPUs and combine them into one dictionary
        result = {u'ydomain': 1,
                  u'xdomain': dt*(steps-ignored_steps),
                  u'dt': dt*10,
                  u'data': {}}

        for k, lpu in lpus.iteritems():
            with h5py.File(lpu['output_file']) as output_file:
                uids = output_file['V']['uids'][:]
                output_array = output_file['V']['data'][:]
                for i, item in enumerate(uids):
                    output = output_array[int(ignored_steps/10):,i:i+1]
                    tmp = output.max()-output.min()
                    if tmp <= 0.01: #mV
                        output = (output - output.min()) + 0.5
                    else:
                        output = (output - output.min())/tmp*0.9+0.1
                    result['data'][item] = np.hstack(
                        (np.arange(int((steps-ignored_steps)/10)).reshape((-1,1))*dt*10, output)).tolist()

        return inputs, result


class AppSession(ApplicationSession):

    log = Logger()
    
    def onConnect(self):
        self.join(self.config.realm, [u"wampcra"], user)

    def onChallenge(self, challenge):
        if challenge.method == u"wampcra":
            print("WAMP-CRA challenge received: {}".format(challenge))

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
        
        @inlineCallbacks
        def nk_launch_progressive(task, details=None):
            print task['user']
            user_id = str(task['user'])
            res = server.launch(user_id, task)
            
            try:
                res_to_processor = yield self.call(task['forward'], res)
                returnValue(res_to_processor)
            except ApplicationError as e:
                print e
                returnValue(False)
    
        uri = 'ffbo.nk.launch.%s' % str(details.session)
        yield self.register(nk_launch_progressive, uri)
        self.log.info('procedure %s registered' % uri)

        # Listen for ffbo.processor.connected
        @inlineCallbacks
        def register_component():
            self.log.info( "Registering a component")
            # CALL server registration
            try:
                # registered the procedure we would like to call
                res = yield self.call('ffbo.server.register',details.session,'nk','nk_server')
                self.log.info("register new server called with result: {result}",result=res)

            except ApplicationError as e:
                if e.error != 'wamp.error.no_such_procedure':
                    raise e

        yield self.subscribe(register_component, 'ffbo.processor.connected')
        self.log.info("subscribed to topic 'ffbo.processor.connected'")

        register_component()

if __name__ == '__main__':
    
    import neurokernel.mpi_relaunch
    
    from twisted.internet._sslverify import OpenSSLCertificateAuthorities
    from twisted.internet.ssl import CertificateOptions
    import OpenSSL.crypto
    
    # parse command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output.')
    parser.add_argument('--url', dest='url', type=six.text_type, default=url,
                        help='The router URL (defaults to value from config.py)')
    parser.add_argument('--realm', dest='realm', type=six.text_type, default=realm,
                        help='The realm to join (defaults to value from config.py).')
    parser.add_argument('--ca_cert', dest='ca_cert_file', type=six.text_type,
                        default=ca_cert_file,
                        help='Root CA PEM certificate file (defaults to value from config.py).')
    parser.add_argument('--int_cert', dest='intermediate_cert_file', type=six.text_type,
                        default=intermediate_cert_file,
                        help='Intermediate PEM certificate file (defaults to value from config.py).')
    parser.add_argument('--no-ssl', dest='ssl', action='store_false')
    parser.add_argument('--no-auth', dest='authentication', action='store_false')
    parser.set_defaults(ssl=ssl)
    parser.set_defaults(authentication=authentication)
    parser.set_defaults(debug=debug)
    
    args = parser.parse_args()

    # start logging
    if args.debug:
        txaio.start_logging(level='debug')
    else:
        txaio.start_logging(level='info')

   # any extra info we want to forward to our ClientSession (in self.config.extra)
    extra = {'auth': args.authentication}

    if args.ssl:
        st_cert=open(args.ca_cert_file, 'rt').read()
        c=OpenSSL.crypto
        ca_cert=c.load_certificate(c.FILETYPE_PEM, st_cert)
        
        st_cert=open(args.intermediate_cert_file, 'rt').read()
        intermediate_cert=c.load_certificate(c.FILETYPE_PEM, st_cert)
        
        certs = OpenSSLCertificateAuthorities([ca_cert, intermediate_cert])
        ssl_con = CertificateOptions(trustRoot=certs)

        # now actually run a WAMP client using our session class ClientSession
        runner = ApplicationRunner(url=args.url, realm=args.realm, extra=extra, ssl=ssl_con)

    else:
        # now actually run a WAMP client using our session class ClientSession
        runner = ApplicationRunner(url=args.url, realm=args.realm, extra=extra)

    runner.run(AppSession)
