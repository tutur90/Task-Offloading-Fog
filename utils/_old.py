def main():
    config_path = "./configs/Pakistan/GA/NSGA2.yaml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    logger = Logger(config)
    env = create_env(config)
    
    valid_size = config["training"].get("valid_size", 0.2)
    
    # Load train and test datasets.
    train_data = pd.read_csv(f"eval/benchmarks/{config['env']['dataset']}/data/{config['env']['flag']}/trainset.csv")
    train_data, valid_data = train_data.iloc[:int(len(train_data)*(1-valid_size))], train_data.iloc[int(len(train_data)*(1-valid_size)):]
    valid_data["GenerationTime"] = valid_data["GenerationTime"] - valid_data["GenerationTime"].min()
    
    test_data = pd.read_csv(f"eval/benchmarks/{config['env']['dataset']}/data/{config['env']['flag']}/testset.csv") 

    if config["policy"] == "NPGA":
        policy = NPGAPolicy(env, config)
    if config["policy"] == "NSGA2":
        policy = NSGA2Policy(env, config)
        
    best_score = np.inf
    best_epoch = 0
    best_individual = None
    
    # print(policy.population[0])
    
    # print(torch.load(
        
    #     "/home/arthur/Documents/Cours/3A/ResearchProject/Task-Offloading-Fog/logs.old2/Pakistan/Tuple30K/MLP/num_epochs_15_batch_size_256_lr_0.005_10/DQRL/checkpoints/checkpoint_epoch_8.pt"
    # )
    #       )
    
    
    # Training and testing loop.
    for epoch in range(config["training"]["num_epochs"]):
        logger.update_epoch(epoch)
        
        # Training phase.
        logger.update_mode('Training')
        tr_fitness = run_epoch(config, policy, train_data, train=True)
        SR, L, E, score = tr_fitness[np.argmin(np.array(tr_fitness)[:, 3])]
        update_metrics(logger, env, config, metrics=(SR, L, E, score))

        

        # Validation phase.
        logger.update_mode('Validation')
        fitness = run_epoch(config, policy, valid_data, train=False)
        best_epoch_individual = np.argmin(np.array(fitness)[:, 3])
        SR, L, E, score = fitness[best_epoch_individual]
        update_metrics(logger, env, config, metrics=(SR, L, E, score))
        env.close()



        if score < best_score:
            best_score = score
            best_epoch = epoch

            best_individual = policy.individuals()[best_epoch_individual]

        
        
        # Plot Pareto for this epoch.
        plot_pareto(fitness, logger.log_dir, epoch=epoch)
        
    # Load the best individual.
    
    print(f"Best individual found at epoch {best_epoch} with score {best_score}")
    
    policy.population = [best_individual]
    policy.individuals = lambda: [best_individual]
    
        
    ## Final evaluation on test data.
    logger.update_mode('Testing')
    fitness = run_epoch(config, policy, test_data, train=False)
    SR, L, E, score = fitness[np.argmin(np.array(fitness)[:, 3])]
    update_metrics(logger, env, config, metrics=(SR, L, E, score))

    logger.plot()
    logger.save_csv()
    
    vis_stats = VisStats(save_path=logger.log_dir)
    vis_stats.vis(env)
    

    # Plot final Pareto frontiers.
    plot_pareto(fitness, logger.log_dir)
    
    logger.close()
    env.close()

if __name__ == '__main__':
    main()