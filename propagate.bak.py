def propagate(cd : m.CountData,
              thrs : float = 5e-4,
              dt : float = 0.1,
              stopafter : int = 10e10,
              normalize : bool = True,
              diffusion_rate : int = 1,
              shuffle : bool = False,
              )-> np.ndarray:

    D = diffusion_rate
    n_genes =  cd.G 
    times = np.zeros(n_genes)
    n_saturated = cd.saturated.shape[0]

    if n_saturated < 1:
        print("[ERROR] : No Saturated spots")
        sys.exit(-1)
    else:
        print("[INFO] : {} Saturated Spots".format(n_saturated))

    if normalize:
        rowMax = np.mean(cd.cnt.values,axis = 1).reshape(-1,1)
        ncnt = np.divide(cd.cnt.values,rowMax,where = rowMax > 0)
        colMax = np.max(ncnt,axis = 0).reshape(1,-1)
        ncnt = np.divide(cd.cnt.values,colMax,where = colMax > 0)


    else:
        ncnt = cd.cnt.values

    if shuffle:
        shf = np.random.permutation(ncnt.shape[0])
        ncnt = ncnt[shf,:]
        iterable = range(n_genes)
    else:
        iterable = tqdm(range(n_genes))
    
    # Get neighbour indices
    nidx = cd.get_allnbr_idx(cd.saturated)

    # Propagate in time
    # Parallel(n_jobs=8)(delayed(stepping)(gene,ncnt,cd,nidx,times,thrs,D,dt,stopafter) for gene in iterable)

    for gene in iterable:
        conc = ncnt[:,gene].astype(float)
        maxDelta = np.inf
        time  = 0
        while maxDelta > thrs and conc[cd.saturated].sum() > 0:
            if time / dt > stopafter:
                genename = cd.cnt.columns[gene]
                print("WARNING : Gene : {} did not convege".format(genename))
                break

            time +=dt

            d2 = cd.laplacian(conc[cd.saturated],
                              conc[nidx],
                              cd.h[cd.saturated])
            dcdt = D*d2
            conc[cd.saturated] = conc[cd.saturated] +  dcdt*dt 

            conc[conc < 0] = 0
            times[gene] = time

            maxDelta = np.max(np.abs(dcdt))

    return times
