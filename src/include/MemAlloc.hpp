

template <typename Species, typename Cell, typename Input>
void MemAlloc(Species& sp, Cell& ce, const Input& in) {
  
  ce.ng_in  = in.ng_in;
  ce.ng_al  = in.ng;
  ce.nl     = in.nl;
  ce.nr     = in.nr;
  ce.ng1    = in.ng1;
  ce.n_cell = in.ncell;

  ce.dt  = in.dt;

  ce.a      = in.a;
  ce.h      = in.h;
  ce.b      = in.b;
  ce.L      = in.L;

  ce.eden.resize(ce.ng1);
  ce.eden_buf.resize(ce.ng1);
  for (int s=0; s<Species::n_species; ++s) {
    sp[s].np   = in.np[s];
    sp[s].q    = in.q[s];
    sp[s].m    = in.m[s];
    sp[s].qdm  = in.q[s]/in.m[s];
    sp[s].w    = in.weight[s];
    ce.edens[s].resize(ce.ng1);
    for (int ix=0; ix<Cell::dim_x; ++ix) 
      sp[s].x[ix].resize(in.np[s]);
    for (int iv=0; iv<Cell::dim_v; ++iv) { 
      sp[s].v[iv].resize(in.np[s]);
      sp[s].E[iv].resize(in.np[s]);
      sp[s].B[iv].resize(in.np[s]);
      ce.jsfield[s][iv].resize(ce.ng1);
    }
  }

  for (int iv=0; iv<Cell::dim_v; ++iv) {
    ce.efield[iv].resize(ce.ng1);
    ce.bfield[iv].resize(ce.ng1);
    ce.jfield[iv].resize(ce.ng1);
  }

};
