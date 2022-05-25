import random
import time
import cPickle
import numpy as np
import subprocess as sub
from functools import partial

from evaluation import evaluate_all
from selection import pareto_selection
from mutation import create_new_children_through_mutation
from logging import PrintLog, initialize_folders, make_gen_directories, write_gen_stats
from diversity import diversity_check_in_children


#Defines some of the objects will be used in run 
#evaluate = evaluate_all : means that all objects of this class perform evaluation when self.evaluate
class Optimizer(object):
    def __init__(self, sim, env, evaluation_func=evaluate_all):
        self.sim = sim
        self.env = env
        if not isinstance(env, list):
            self.env = [env]
        self.evaluate = evaluation_func 
        self.curr_env_idx = 0
        self.start_time = None

    def elapsed_time(self, units="s"):
        if self.start_time is None:
            self.start_time = time.time()
        s = time.time() - self.start_time
        if units == "s":
            return s
        elif units == "m":
            return s / 60.0
        elif units == "h":
            return s / 3600.0

#This function saves the pickle files
    def save_checkpoint(self, directory, gen):
        random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        data = [self, random_state, numpy_random_state]
        with open('{0}/pickledPops/Gen_{1}.pickle'.format(directory, gen), 'wb') as handle:
            cPickle.dump(data, handle, protocol=cPickle.HIGHEST_PROTOCOL)

    def run(self, *args, **kwargs):
        raise NotImplementedError

#run function here
class PopulationBasedOptimizerDiversityAlternation(Optimizer):
    def __init__(self, sim, env, pop, selection_func, mutation_func,diversity_check_function = None,diversity_alternation_prob = 0.5):
        Optimizer.__init__(self, sim, env)
        self.pop = pop
        self.select = selection_func
        self.mutate = mutation_func
        self.num_env_cycles = 0
        self.autosuspended = False
        self.max_gens = None
        self.directory = None
        self.name = None
        self.num_random_inds = 0
        self.diversity_check = diversity_check_function
        self.diversity_alternation_prob = diversity_alternation_prob


    # Function used when you change (in cycles) the environment beeing used during the simulation
    def update_env(self):
        if self.num_env_cycles > 0:
            switch_every = self.max_gens / float(self.num_env_cycles)
            self.curr_env_idx = int(self.pop.gen / switch_every % len(self.env))
            print " Using environment {0} of {1}".format(self.curr_env_idx+1, len(self.env))

    def run(self, max_hours_runtime=29, max_gens=350, num_random_individuals=1, num_env_cycles=0,
            directory="tests_data", name="TestRun",
            max_eval_time=60, time_to_try_again=30,
            checkpoint_every=100, save_vxa_every=100, save_pareto=True,
            save_nets=False, save_lineages=False, continued_from_checkpoint=False,
            batch_size=None, update_survivors_age=True,
            max_fitness=None,add_new_ind_fixed_shape = False,fixed_shape = None,TO_ENV = None):

        if self.autosuspended:
            sub.call("rm %s/AUTOSUSPENDED" % directory, shell=True) #sub.call("rm %s/AUTOSUSPENDED" % directory, shell=True)


        self.autosuspended = False
        self.max_gens = max_gens  # can add additional gens through checkpointing
        self.directory = directory #updates if the directory changes

        #Class that init a dictionary that contain important times
        print_log = PrintLog() #here you have the default times: start and last_call
        print_log.add_timer("evaluation") #add "evaluation" time in the dict list
        self.start_time = print_log.timers["start"]  # sync start time with logging

        # sub.call("clear", shell=True)

        #If you are starting from generation zero
        if not continued_from_checkpoint:  # generation zero
            self.directory = directory 
            self.name = name 
            self.num_random_inds = num_random_individuals #parameter about the number of new random individuals added in each generations. typical example = 1
            self.num_env_cycles = num_env_cycles #parameter to change environment. typical example = 1 (same environment always)

            #Function used to create the folders where the simulations outputs will be saved: 
            initialize_folders(self.pop, self.directory, self.name, save_nets, save_lineages=save_lineages)

            #saves networks and vxa file for each generation if save_vxa_every and save_nets = True
            make_gen_directories(self.pop, self.directory, save_vxa_every, save_nets)

            #Create RUNNING file
            sub.call("touch {}/RUNNING".format(self.directory), shell=True)

            #Evaluate fitness of all individuals of the population in Voxelyze
            #In evaluation, the functions read_voxlyze_results AND write_voxelyze_file are used
            #So it is the place where the contact with Voxelyze is done.
            self.evaluate(self.sim, self.env[self.curr_env_idx], self.pop, print_log, save_vxa_every, self.directory,
                          self.name, max_eval_time, time_to_try_again, save_lineages, batch_size)
            
                 
            self.select(self.pop)  # only produces dominated_by stats, no selection happening (population not replaced)
            write_gen_stats(self.pop, self.directory, self.name, save_vxa_every, save_pareto, save_nets,
                            save_lineages=save_lineages)

        while self.pop.gen < max_gens:

            if self.pop.gen % checkpoint_every == 0: #CHECK_POINT_EVERY = 1 saves always
                print_log.message("Saving checkpoint at generation {0}".format(self.pop.gen+1), timer_name="start")
                self.save_checkpoint(self.directory, self.pop.gen)

            if self.elapsed_time(units="h") > max_hours_runtime or self.pop.best_fit_so_far == max_fitness:
                self.autosuspended = True
                print_log.message("Autosuspending at generation {0}".format(self.pop.gen+1), timer_name="start")
                self.save_checkpoint(self.directory, self.pop.gen)
                sub.call("touch {0}/AUTOSUSPENDED && rm {0}/RUNNING".format(self.directory), shell=True)
                break

            self.pop.gen += 1
            print_log.message("Creating folders structure for this generation")
            make_gen_directories(self.pop, self.directory, save_vxa_every, save_nets)

            # update ages
            self.pop.update_ages(update_survivors_age)

            P = self.diversity_alternation_prob #probability of using the diversity method to generate the new children 
            DIVERSITY_METHOD = np.random.choice([True,False],p=(P,1-P))
            if DIVERSITY_METHOD:
                LOW_DIVERSITY = True
                diverse_children = []
                COUNT_TRIALS = 0

                while LOW_DIVERSITY:
                    COUNT_TRIALS += 1
                    # mutation
                    print_log.message("Mutation starts")
                    new_children = self.mutate(self.pop, print_log=print_log)
                    print_log.message("Mutation ends: successfully generated %d new children." % (len(new_children)))

                    diverse_children = self.diversity_check(self.pop,new_children,diverse_children)

                    if len(diverse_children) >= self.pop.pop_size:
                        LOW_DIVERSITY = False
                    
                    print_log.message("{0} new diverse children until now".format(len(diverse_children)))
                
                    if COUNT_TRIALS >= 25:
                        print_log.message("Diversity method not working. Using normal method. Mutation starts")
                        diverse_children = self.mutate(self.pop, print_log=print_log)
                        print_log.message("Mutation ends: successfully generated %d new children." % (len(new_children)))
                        break

                print_log.message("Finish generating diverse new children! It took {0} trials".format(COUNT_TRIALS))

                # combine children and parents for selection
                print_log.message("Now creating new population")
                self.pop.append(diverse_children) #Add the new children to population

            else:
                #mutation
                print_log.message("Mutation starts")
                new_children = self.mutate(self.pop, print_log=print_log)
                print_log.message("Mutation ends: successfully generated %d new children." % (len(new_children)))

                # combine children and parents for selection
                print_log.message("Now creating new population")
                self.pop.append(new_children) #Add the new children to population


            if add_new_ind_fixed_shape:
                for _ in range(self.num_random_inds):
                    print_log.message("Random individual added to population")
                    self.pop.add_random_individual(add_new_ind_fixed_shape = True,fixed_shape = fixed_shape,TO_ENV = TO_ENV) #add new random individual to population
                print_log.message("New population size is %d" % len(self.pop))
            else:
                for _ in range(self.num_random_inds):
                    print_log.message("Random individual added to population")
                    self.pop.add_random_individual() #add new random individual to population
                print_log.message("New population size is %d" % len(self.pop))

            

            # evaluate fitness
            print_log.message("Starting fitness evaluation", timer_name="start")
            print_log.reset_timer("evaluation")
            self.update_env()
            self.evaluate(self.sim, self.env[self.curr_env_idx], self.pop, print_log, save_vxa_every, self.directory,
                          self.name, max_eval_time, time_to_try_again, save_lineages, batch_size)
            print_log.message("Fitness evaluation finished", timer_name="evaluation")  # record total eval time in log

            # perform selection by pareto fronts
            new_population = self.select(self.pop)
            
            # print population to stdout and save all individual data
            print_log.message("Saving statistics")
            write_gen_stats(self.pop, self.directory, self.name, save_vxa_every, save_pareto, save_nets,
                            save_lineages=save_lineages)

            # replace population with selection
            self.pop.individuals = new_population
            print_log.message("Population size reduced to %d" % len(self.pop))

        if not self.autosuspended:  # print end of run stats
            print_log.message("Finished {0} generations".format(self.pop.gen + 1))
            print_log.message("DONE!", timer_name="start")
            sub.call("touch {0}/RUN_FINISHED && rm {0}/RUNNING".format(self.directory), shell=True)


class ParetoOptimizationDiversityChildrenAlternation(PopulationBasedOptimizerDiversityAlternation):
    def __init__(self, sim, env, pop,similarity_threshold_children,diversity_alternation_prob,mut_net_probs):
        PopulationBasedOptimizerDiversityAlternation.__init__(self, sim, env, pop, pareto_selection,
                                    partial(create_new_children_through_mutation, mutate_network_probs=mut_net_probs), 
                                    diversity_check_function = partial(diversity_check_in_children,similarity_threshold_children=similarity_threshold_children),
                                    diversity_alternation_prob = diversity_alternation_prob