from dataclasses import dataclass
import ROOT as r
r.gSystem.Load('libRooFit')
import numpy as np
from rebsmearv2.helpers.paths import rebsmear_path

@dataclass(frozen=False)
class Jet():
    pt: float
    eta: float
    phi: float
    jetid: float
    sieie: float
    sipip: float
    hfcss: float
    px: float = 0
    py: float = 0
    pz: float = 0

    def __post_init__(self):
        object.__setattr__(self, 'px', np.cos(self.phi) * self.pt)
        object.__setattr__(self, 'py', np.sin(self.phi) * self.pt)
        object.__setattr__(self, 'pz', np.sinh(self.eta) * self.pt)

    def set_pt(self, ptvalue):
        self.pt = ptvalue
        # Update the Cartesian components of the momenta after the transverse momentum gets an update
        self.px = np.cos(self.phi) * ptvalue
        self.py = np.sin(self.phi) * ptvalue
        self.pz = np.sinh(self.eta) * ptvalue

class NamingMixin():

    def _name_jet_momentum_pdf(self,direction, index):
        return f"momentum_pdf_{direction}_{index}"

    def _name_gen_momentum_var(self,direction, index):
        return f"gen_{direction}_{index}"

    def _name_reco_momentum_var(self,direction, index):
        return f"reco_{direction}_{index}"

    def _name_jet_resolution_var(self,direction, index):
        return f"sigma_{direction}_{index}"

    def _name_partial_gen_htmiss_variable(self,direction):
        return f"gen_htmiss_{direction}"

    def _name_total_gen_ht_variable(self):
        return "gen_ht"

    def _name_total_gen_htmiss_variable(self):
        return f"gen_htmiss_pt"

    def _name_combined_momentum_pdf(self):
        return f"momentum_pdf_total"

    def _name_likelihood(self):
        return "likelihood"

    def _name_total_prior_pdf(self):
        return 'total_prior_pdf'

    def _name_metadata_njets_variable(self):
        return 'njets'


def make_RooArgList(items):
    l = r.RooArgList()
    for item in items:
        l.add(item)
    return l


class HistoSF2D():
    def __init__(self, histogram):
        assert(histogram)
        self._histogram = histogram
        self._init_boundaries()

    def _init_boundaries(self):
        bin_bottom_left = self._histogram.GetBin(1,1)
        nbins_x = self._histogram.GetNbinsX()
        nbins_y = self._histogram.GetNbinsY()
        self._xmin = self._histogram.GetXaxis().GetBinCenter(1)
        self._xmax = self._histogram.GetXaxis().GetBinCenter(nbins_x-1)
        self._ymin = self._histogram.GetYaxis().GetBinCenter(1)
        self._ymax = self._histogram.GetYaxis().GetBinCenter(nbins_y-1)


    def _apply_limit(self, value, low, high):
        return max(low, min(value, high))

    def evaluate(self,x,y):
        x = self._apply_limit(x, self._xmin, self._xmax)
        y = self._apply_limit(y, self._ymin, self._ymax)

        bin_id = self._histogram.FindBin(x,y)
        return self._histogram.GetBinContent(bin_id)

    def __call__(self,x,y):
        return self.evaluate(x,y)

class ConstantJER():
    def __init__(self, constant_value):
        self.constant_value = constant_value

    def evaluate(self, x, y):
        return self.constant_value

    def __call__(self,x,y):
        return self.evaluate(x,y)

class JERLookup():
    def __init__(self):
        pass

    def from_th1(self, filepath, histogram_name):
        '''Set input TH1 histogram for JER evaluation.'''
        f = r.TFile(filepath)
        if not f:
            raise IOError(f"Could not open file: '{filepath}'")

        h = f.Get(histogram_name)
        if not f:
            raise IOError(f"Could not load histogram: '{histogram_name}'")

        h.SetDirectory(0)
        self._evaluator = HistoSF2D(h)

    def from_constant(self, constant_value):
        '''Set constant sigma value for JER evaluation.'''
        self._evaluator = ConstantJER(constant_value)

    def get_jer(self, pt, eta):
        return self._evaluator(pt, np.abs(eta))

class RebalanceWSFactory(NamingMixin):
    '''
    Factory class for a RooWorkspace used for rebalancing fits.

    The class is initiated based on a list of jets.

    jets = [Jet(pt, eta, phi) for pt, eta, phi in ...]
    factory = RebalanceWSFactory(jets)
    factory.build()
    '''
    def __init__(self,jets):
        self.jets = jets
        self.njets = len(jets)
        self.ws = r.RooWorkspace()
        self._wsimp = getattr(self.ws, 'import')
        self._jer_evaluator = None
        self._directions = 'pt','phi'
    def set_jer_evaluator(self,jer_evaluator):
        self._jer_evaluator = jer_evaluator

    def get_ws(self):
        return self.ws

    def get_jet(self, index):
        return self.jets[index]

    def build(self):
        '''
        Defines all ingredients for the fit model.
        '''
        self._build_metadata()
        self._build_all_jets()
        self._build_combined_momentum_pdf()
        self._build_priors()
        self._build_likelihood()
        self._build_negative_log_likelihood()

    def _build_metadata(self):
        self._build_metadata_njets()

    def _build_metadata_njets(self):
        njets_variable_name = self._name_metadata_njets_variable()
        njets_variable = r.RooRealVar(
                                      njets_variable_name,
                                      njets_variable_name,
                                      self.njets
                                      )
        self._wsimp(njets_variable)

    def _name_negative_log_likelihood(self):
        return "nll"

    def _build_negative_log_likelihood(self):
        likelihood_name = self._name_likelihood()
        likelihood_function = self.ws.function(likelihood_name)
        nll_name = self._name_negative_log_likelihood()
        expression = f"- log ({likelihood_name})"
        nll = r.RooGenericPdf(
            nll_name,
            nll_name,
            expression,
            r.RooArgList(likelihood_function)
        )
        self._wsimp(nll)


    def _build_likelihood(self):
        partial_pdf_names = [
            self._name_total_prior_pdf(),
            self._name_combined_momentum_pdf()
        ]
        partial_pdfs = [self.ws.function(x) for x in partial_pdf_names]

        expression = '*'.join(partial_pdf_names)
        likelihood_name = self._name_likelihood()
        likelihood = r.RooGenericPdf(
            likelihood_name,
            likelihood_name,
            expression,
            r.RooArgList(*partial_pdfs)
        )
        self._wsimp(likelihood)

    def _build_gen_ht_variable(self):
        expression_parts = []
        variables = []
        for index in range(self.njets):
            if self._directions == ('pt','phi'):
                name = self._name_gen_momentum_var('pt', index)
                expression_parts.append(name)

                var = self.ws.var(name)
                variables.append(var)
            else:
                name_x = self._name_gen_momentum_var('px', index)
                name_y = self._name_gen_momentum_var('py', index)
                expression_parts.append(f"sqrt({name_x}**2 + {name_y}**2)")

                var_x = self.ws.var(name_x)
                var_y = self.ws.var(name_y)
                variables.append(var_x)
                variables.append(var_y)

        expression = '+'.join(expression_parts)
        name_ht = self._name_total_gen_ht_variable()
        ht_variable = r.RooFormulaVar(
            name_ht,
            expression,
            make_RooArgList(variables)
        )
        self._wsimp(ht_variable)


    def _build_gen_htmiss_variables(self):
        if self._directions==('pt','phi'):
            self._build_gen_htmiss_xy_variable_from_pt_phi()
        else:
            self._build_gen_htmiss_xy_variable_from_px_py()
        self._build_derived_gen_htmiss_pt_phi_variables()

    def _build_gen_htmiss_xy_variable_from_pt_phi(self):
        '''
        Creates x and y variables for htmiss based on jet pt, phi inputs.
        '''
        expression_parts_x, expression_parts_y = [], []
        variables = []
        for index in range(self.njets):
            var_name_pt = self._name_gen_momentum_var('pt', index)
            var_name_phi = self._name_gen_momentum_var('phi', index)
            variables.append(self.ws.var(var_name_phi))
            variables.append(self.ws.var(var_name_pt))
            expression_parts_x.append(f'{var_name_pt} * cos({var_name_phi})')
            expression_parts_y.append(f'{var_name_pt} * sin({var_name_phi})')
        args = make_RooArgList(variables)

        name_px = self._name_partial_gen_htmiss_variable("px")
        expression_x = '+'.join(expression_parts_x)
        htmiss_px_variable = r.RooFormulaVar(
            name_px,
            expression_x,
            args
        )
        self._wsimp(htmiss_px_variable)

        name_py = self._name_partial_gen_htmiss_variable("py")
        expression_y = '+'.join(expression_parts_y)
        htmiss_py_variable = r.RooFormulaVar(
            name_py,
            expression_y,
            args
        )
        self._wsimp(htmiss_py_variable)

    def _build_gen_htmiss_xy_variable_from_px_py(self):
        '''
        Creates x and y variables for htmiss based on jet px, py inputs.
        '''
        for direction in self._directions:
            self._build_gen_htmiss_xy_variable_single_direction(direction)

    def _build_gen_htmiss_xy_variable_single_direction(self, direction):
        '''
        Create gen HTmiss variable in x or y direction.
        '''
        expression_parts = []
        variables = []
        for index in range(self.njets):
            var_name = self._name_gen_momentum_var(direction, index)
            variables.append(self.ws.var(var_name))
            expression_parts.append(var_name)
        expression = '+'.join(expression_parts)

        name = self._name_partial_gen_htmiss_variable(direction)
        htmiss_partial_variable = r.RooFormulaVar(
            name,
            expression,
            make_RooArgList(variables)
        )

    def _build_derived_gen_htmiss_pt_phi_variables(self):
        '''
        Create derived HTmiss pt, phi coordinates based on existing px, py.

        This is just a basis transformation for convenience in future use.
        '''
        name_px = self._name_partial_gen_htmiss_variable("px")
        htmiss_px_variable = self.ws.function(name_px)
        name_py = self._name_partial_gen_htmiss_variable("py")
        htmiss_py_variable = self.ws.function(name_py)
        htmiss_pt_variable = r.RooFormulaVar(
            self._name_partial_gen_htmiss_variable("pt"),
            f'sqrt({name_px}**2 + {name_py}**2)',
            r.RooArgList(htmiss_px_variable, htmiss_py_variable)
        )
        self._wsimp(htmiss_pt_variable)


    def _build_total_gen_htmiss_variable(self):
        partial_htmiss_variable_names = [self._name_partial_gen_htmiss_variable(direction) for direction in self._directions]
        partial_htmiss_variable = [self.ws.function(x) for x in partial_htmiss_variable_names]
        expression = f"sqrt({'+'.join([f'{X}**2' for X in partial_htmiss_variable_names])})"
        total_htmiss_variable = r.RooFormulaVar(
            self._name_total_gen_htmiss_variable(),
            expression,
            make_RooArgList(partial_htmiss_variable)
        )
        self._wsimp(total_htmiss_variable)

    def _build_partial_gen_htmiss_variable(self, direction):
        momentum_variable_names = self._expand_naming(self._name_gen_momentum_var, directions=[direction])
        momentum_variables = [self.ws.var(x) for x  in momentum_variable_names]
        htmiss_variable_name = self._name_partial_gen_htmiss_variable(direction)
        expression = '+'.join(momentum_variable_names)
        partial_htmiss_variable = r.RooFormulaVar(
            htmiss_variable_name,
            expression,
            make_RooArgList(momentum_variables)
        )
        self._wsimp(partial_htmiss_variable)

    def _name_total_gen_htmiss_prior_pdf(self):
        return 'gen_htmiss_prior_pdf'

    def _name_total_gen_htmiss_prior_slope(self):
        return 'gen_htmiss_prior_slope'

    def _get_gen_htmiss_prior_file(self):
        rfile = rebsmear_path('./input/htmiss_prior.root')
        return r.TFile(rfile)

    def _name_ht_bin(self, gen_ht):
        '''Given the GEN-HT of the event, figure out which HT bin it corresponds to.'''
        # Binning of the prior in terms of HT
        htbins = [100, 300, 500, 700, 900, 1300, 2000, 5000]
        for idx in range(len(htbins)-1):
            lo = htbins[idx]
            hi = htbins[idx+1]

            if (lo <= gen_ht) and (gen_ht <= hi):
                return f'{lo:.0f}_to_{hi:.0f}'

        raise RuntimeError(f'Could not figure out the HT bin for HT: {gen_ht:.3f}')

    def _get_prior_histogram(self, prior_input_file):
        # Get the HT variable from workspace
        gen_ht = self.ws.function(self._name_total_gen_ht_variable())
        ht_bin_for_event = self._name_ht_bin(gen_ht.evaluate())

        # Get the right prior histogram from the input file and return the histogram
        for key in prior_input_file.GetListOfKeys():
            hist = key.ReadObj()
            histname = hist.GetName()
            if '2018' in histname:
                continue
            if ht_bin_for_event not in histname:
                continue

            return hist

    def _build_gen_htmiss_prior_roohistpdf(self, th1):
        '''Do the TH1 -> RooDataHist -> RooHistPdf conversion for the prior PDF. Saves the final RooHistPdf into the workspace.'''
        htmiss_variable = self.ws.function(self._name_partial_gen_htmiss_variable(direction='pt'))

        # We need to provide at least one RooRealVar (primitive variable) into the RooDataHist and RooHistPdf
        # Providing only derived quantities do not work in the datahist/pdf constructors
        # To achieve that, we define this dummy RooRealVar and evaluate it at the initial value of HTmiss.
        dummy_htmiss_variable = r.RooRealVar(
            'dummy_gen_htmiss_pt',
            'dummy_gen_htmiss_pt',
            htmiss_variable.evaluate()
        )

        # Name for datahist, not important since the datahist is just an intermediate step anyway
        th1name = th1.GetName()

        datahist = r.RooDataHist(th1name, th1name,
                        r.RooArgList(dummy_htmiss_variable),
                        th1
                    )

        prior_pdf_name = self._name_total_gen_htmiss_prior_pdf()

        prior_pdf = r.RooHistPdf(prior_pdf_name,
                        prior_pdf_name,
                        r.RooArgList(htmiss_variable),
                        r.RooArgList(dummy_htmiss_variable),
                        datahist
                    )

        # Linear interpolation
        prior_pdf.setInterpolationOrder(1)

        self._wsimp(prior_pdf)

    def _build_gen_htmiss_prior(self):
        '''Build gen HTmiss prior, extracted from the source file.'''
        # Get the source file for prior distributions
        prior_input_file = self._get_gen_htmiss_prior_file()

        # For the event at hand, do the following:
        # 1. Get the histogram corresponding to the HT bin, based on GEN-HT of event
        hist = self._get_prior_histogram(prior_input_file)

        # 2. Convert it into a RooDataHist and finally a RooHistPDF
        # 3. Save the RooHistPDF to the workspace
        self._build_gen_htmiss_prior_roohistpdf(hist)

    def _build_total_prior(self):
        pdf_name = self._name_total_prior_pdf()
        partial_prior_pdf_names = [self._name_total_gen_htmiss_prior_pdf()]
        partial_prior_pdfs = [self.ws.function(x) for x in partial_prior_pdf_names]
        expression = '*'.join(partial_prior_pdf_names)
        total_prior_pdf = r.RooGenericPdf(
            pdf_name,
            pdf_name,
            expression,
            r.RooArgList(*partial_prior_pdfs)
        )
        self._wsimp(total_prior_pdf)

    def _build_priors(self):
        self._build_gen_ht_variable()
        self._build_gen_htmiss_variables()
        self._build_gen_htmiss_prior()
        self._build_total_prior()

    def _build_all_jets(self):
        '''
        Defines gen->reco PDFs for all jets.
        '''
        for n in range(self.njets):
            self._build_single_jet(n)

    def _expand_naming(self, naming_function, directions=None, indices=None):
        '''
        Creates a list of names for all jets given a pattern defined by the naming function.
        '''
        if directions is None:
            directions = self._directions
        if indices is None:
            indices = range(self.njets)
        return [naming_function(direction, index) for direction in directions for index in indices]

    def _variable_limits(self, direction, central_value):
        if direction == 'phi':
            return (None, None)
        else:
            lim = max(2*abs(central_value), 100)
            return (-lim, lim)

    def _build_jet_eta(self, index):
        '''
        Defines RooRelVars for jet eta.
        '''
        jet = self.get_jet(index)
        central_value = getattr(jet, 'eta')

        name_reco_var = self._name_reco_momentum_var('eta', index)
        reco_var = r.RooRealVar(
                                name_reco_var,
                                name_reco_var,
                                central_value
                                )
        self._wsimp(reco_var)

    def _build_single_jet_momentum_vars(self, direction, index):
        '''
        Defines RooRealVars for gen and reco momenta for a given momentum direction and jet index.
        '''
        jet = self.get_jet(index)

        central_value = getattr(jet, direction)

        args = [central_value]

        # Limits depend on the direction,
        # e.g. phi maybe assumed constant
        # Only add limits to constructor if
        # the variable is NOT constant
        limits = self._variable_limits(direction, central_value)
        if not any([x is None for x in limits]):
            args.extend(limits)

        name_gen_var = self._name_gen_momentum_var(direction, index)
        gen_var = r.RooRealVar(
                            name_gen_var,
                            name_gen_var,
                            *args
                            )
        self._wsimp(gen_var)


        name_reco_var = self._name_reco_momentum_var(direction, index)
        reco_var = r.RooRealVar(
                                name_reco_var,
                                name_reco_var,
                                central_value
                                )
        self._wsimp(reco_var)

        return (gen_var, reco_var)

    def _resolution(self, index, direction):
        '''
        The jet resolution in a given direction for given jet index in GeV.
        '''
        jet = self.get_jet(index)
        sigma = self._jer_evaluator.get_jer(jet.pt, jet.eta)

        return sigma * getattr(jet, direction)

    def _build_single_jet(self, index):
        '''
        Defines variables and PDFs for a single jet index.
        '''
        for direction in self._directions:
            gen_var, reco_var = self._build_single_jet_momentum_vars(direction, index)
            self._build_single_jet_momentum_pdf(gen_var, reco_var, direction, index)

        self._build_jet_eta(index)

    def _build_single_jet_momentum_pdf(self, gen_var, reco_var, direction, index):
        '''
        Defines the PDF(reco | gen), i.e. the probability representing the agreement
        between gen and reco momentum for a given direction and jet index.
        '''

        if gen_var.isConstant():
            return
        else:
            pdf_name = self._name_jet_momentum_pdf(direction, index)
            sigma = self._resolution(index, direction)
            resolution_name = self._name_jet_resolution_var(direction, index)
            resolution_var = r.RooRealVar(
                                    resolution_name,
                                    resolution_name,
                                    sigma
                                    )
            self._wsimp(resolution_var)

            momentum_pdf = r.RooGaussian(
                                        pdf_name,
                                        pdf_name,
                                        reco_var,
                                        gen_var,
                                        resolution_var
                                        )

            self._wsimp(momentum_pdf)


    def _build_combined_momentum_pdf(self):
        '''
        Defines the product PDF of all individual jet PDFs.
        '''
        individual_pdf_names = self._expand_naming(self._name_jet_momentum_pdf)

        individual_pdfs, expression_parts = [], []
        for name in individual_pdf_names:
            func = self.ws.function(name)
            if not func:
                continue
            individual_pdfs.append(func)
            expression_parts.append(name)

        pdf_name = self._name_combined_momentum_pdf()
        expression = '*'.join(expression_parts)
        combined_pdf = r.RooGenericPdf(
                    pdf_name,
                    pdf_name,
                    expression,
                    make_RooArgList(individual_pdfs)
                    )

        self._wsimp(combined_pdf)
