/*
 *  MoMEMta: a modular implementation of the Matrix Element Method
 *  Copyright (C) 2016  Universite catholique de Louvain (UCL), Belgium
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// Must be loaded before `momemta/Logging.h`, otherwise there's conflict between usage
// of `log()` and `namespace log`
#include <numeric>
#include <LHAPDF/LHAPDF.h>

#include <momemta/Logging.h>
#include <momemta/MatrixElement.h>
#include <momemta/MatrixElementFactory.h>
#include <momemta/ParameterSet.h>
#include <momemta/Math.h>
#include <momemta/Module.h>
#include <momemta/Types.h>
#include <momemta/Utils.h>

#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TSystem.h"
#include "TObject.h"

#include "lwtnn/LightweightGraph.hh"
#include "lwtnn/parse_json.hh"

/** \brief Compute the integrand: matrix element, PDFs, jacobians
 *
 * ### Summary
 *
 * Evaluate the matrix element, parton density functions, jacobians,
 * phase-space density terms, flux factor, ..., to define the final quantity to be integrated.
 *
 * The matrix element has to be evaluated on all the initial and final particles' 4-momenta.
 * In most cases, a subset of those particles are given by a Block. Blocks produce several
 * equivalent solutions for those particles, and the matrix element, ..., has to be computed on
 * each of those solutions, along with the rest of the particles in the event. For that, you need
 * to use a Looper module, with this module in the execution path.
 *
 * To define the integrand, the quantities produced by this module (one quantity per solution) has to be summed. Use the
 * DoubleSummer module for this purpose.
 *
 * ### In more details
 *
 * In the following, the set of solutions will be indexed by \f$j\f$. Particles produced by the Block will be
 * called 'invisible', while other, uniquely defined particles in the event will be called 'visible'. Since initial
 * state momenta are computed from the whole event, they have the same multiplicity as the 'invisibles' and will therefore
 * also be indexed by \f$j\f$.
 *
 * \warning Keep in mind that the loop describe below is **not** done by this module. A Looper must be used for this.
 *
 * If no invisibles are present and the matrix element only has to be evaluated on the uniquely defined visible particles,
 * this module can still be used to define the integrand: no loop is done in this case, and the index \f$j\f$ can be omitted
 * in the following.
 *
 * As stated above, for each solution \f$j\f$, this modules's ouput is a scalar \f$I\f$:
 * \f[
 *      I_j = \frac{1}{2 x_1^j x_2^j s} \times \left( \sum_{i_1, i_2} \, f(i_1, x_1^j, Q_f^2) \, f(i_2, x_2^j, Q_f^2) \, \left| \mathcal{M}(i_1, i_2, j) \right|^2 \right) \times \left( \prod_{i} \mathcal{J}_i \right)
 * \f]
 * where:
 *    - \f$s\f$ is the hadronic centre-of-mass energy.
 *    - \f$\mathcal{J}_i\f$ are the jacobians.
 *    - \f$x_1^j, x_2^j\f$ are the initial particles' Björken fractions (computed from the entry \f$j\f$ of the initial states given as input).
 *    - \f$f(i, x^j, Q_f^2)\f$ is the PDF of parton flavour \f$i\f$ evaluated on the initial particles' Björken-\f$x\f$ and using factorisation scale \f$Q_f\f$.
 *    - \f$\left| \mathcal{M}(i_1, i_2, j) \right|^2\f$ is the matrix element squared evaluated on all the particles' momenta in the event, for solution \f$j\f$. Along with the PDFs, a sum is done over all the initial parton flavours \f$i_1, i_2\f$ defined by the matrix element.
 *
 * ### Expected parameter sets
 *
 * Some inputs expected by this module are not simple parameters, but sets of parameters and input tags. These are used for:
 *
 *    - matrix element:
 *      - `card` (string): Path to the the matrix element's `param_card.dat` file.
 *
 *    - particles:
 *      - `inputs` (vector(LorentzVector)): Set of particles.
 *      - `ids`: Parameter set used to link the visibles to the matrix element (see below).
 *
 * ### Linking inputs and matrix element
 *
 * The matrix element expects the final-state particles' 4-momenta to be given in a certain order,
 * but it is agnostic as to how the particles are ordered in MoMEMta.
 * It is therefore necessary to specify the index of each input (visible and invisible)
 * particle in the matrix element call. Furthermore, since the matrix element library might define several final states,
 * each input particle's PDG ID has to be set by the user, to ensure the correct final state is retrieved when evaluating the matrix element.
 *
 * To find out the ordering the matrix element expects, it is currently necessary to dig into the matrix element's code.
 * For instance, for the fully leptonic \f$t\overline{t}\f$ example shipped with MoMEMta, the ordering and PDG IDs can be read from [here](https://github.com/MoMEMta/MoMEMta/blob/master/MatrixElements/pp_ttx_fully_leptonic/SubProcesses/P1_Sigma_sm_gg_mupvmbmumvmxbx/P1_Sigma_sm_gg_mupvmbmumvmxbx.cc#L86).
 *
 * In the Lua configuation for this module, the ordering is defined through the `ids` parameter set mentioned above. For instance,
 * ```
 * particles = {
 *     inputs = { 'input::particles/1', 'input::particles/2' },
 *     ids = {
 *         {
 *             me_index = 2,
 *             pdg_id = 11
 *         },
*         {
    *             me_index = 1,
        *             pdg_id = -11
            *         }
            *     }
            * }
            * ```
            * means that the particle vector corresponds to (electron, positron), while the matrix element expects to be given first the positron, then the electron.
            *
            * ### Integration dimension
            *
            * This module requires **0** phase-space point.
            *
            * ### Global Parameters
            *
            *   | Name | Type | %Description |
            *   |------|------|--------------|
            *   | `energy` | double | Hadronic centre-of-mass energy (GeV). |
            *
            * ### Parameters
            *
            *   | Name | Type | %Description |
            *   |------|------|--------------|
            *   | `use_pdf` | bool, default true | Evaluate PDFs and use them in the integrand. |
            *   | `pdf` | string | Name of the LHAPDF set to be used (see [full list](https://lhapdf.hepforge.org/pdfsets.html)). |
            *   | `pdf_scale` | double | Factorisation scale used when evaluating the PDFs. |
            *   | `matrix_element` | string | Name of the matrix element to be used. |
            *   | `matrix_element_parameters` | ParameterSet | Set of parameters passed to the matrix element (see above explanation). |
            *   | `override_parameters` | ParameterSet (optional) | Overrides the value of the ME parameters (usually those specified in the param card) by the ones specified. |
            *
            * ### Inputs
            *
            *   | Name | Type | %Description |
            *   |------|------|--------------|
            *   | `initialState` | vector(vector(LorentzVector)) | Sets of initial parton 4-momenta (one pair per invisibles' solution), typically coming from a BuildInitialState module. |
            *   | `particles` | ParameterSet | Set of parameters defining the particles (see above explanation). |
            *   | `jacobians` | vector(double) | All jacobians defined in the integration (transfer functions, generators, blocks...). |
            *
            * ### Outputs
            *
            *   | Name | Type | %Description |
            *   |------|------|--------------|
            *   | `integrands` | vector(double) | Vector of integrands (one per invisibles' solution). All entries in this vector will be summed by MoMEMta to define the final integrand used by Cuba to compute the integral. |
            *
            * \ingroup modules
            */
            class MatrixElement: public Module {
                struct ParticleId {
                    int64_t pdg_id;
                    int64_t me_index;
                };

                public:

                MatrixElement(PoolPtr pool, const ParameterSet& parameters): Module(pool, parameters.getModuleName()) {

                    sqrt_s = parameters.globalParameters().get<double>("energy");

                    use_pdf = parameters.get<bool>("use_pdf", true);

                    save_ME = parameters.get<bool>("save_ME", false);
                    save_max = parameters.get<int64_t>("save_max", -1);

                    m_partons = get<std::vector<LorentzVector>>(parameters.get<InputTag>("initialState"));

                    const auto& particles_set = parameters.get<ParameterSet>("particles");

                    auto particle_tags = particles_set.get<std::vector<InputTag>>("inputs");
                    for (auto& tag: particle_tags)
                        m_particles.push_back(get<LorentzVector>(tag));

                    const auto& particles_ids_set = particles_set.get<std::vector<ParameterSet>>("ids");
                    for (const auto& s: particles_ids_set) {
                        ParticleId id;
                        id.pdg_id = s.get<int64_t>("pdg_id");
                        id.me_index = s.get<int64_t>("me_index");
                        m_particles_ids.push_back(id);
                    }

                    if (m_particles.size() != m_particles_ids.size()) {
                        LOG(fatal) << "The number of particles ids is not consistent with the number of particles. Did you"
                            " forget some ids?";

                        throw Module::invalid_configuration("Inconsistent number of ids and number of particles");
                    }

                    const auto& jacobians_tags = parameters.get<std::vector<InputTag>>("jacobians", std::vector<InputTag>());
                    for (const auto& tag: jacobians_tags) {
                        m_jacobians.push_back(get<double>(tag));
                    }

                    std::string matrix_element = parameters.get<std::string>("matrix_element");
                    const ParameterSet& matrix_element_configuration = parameters.get<ParameterSet>("matrix_element_parameters");
                    m_ME = MatrixElementFactory::get().create(matrix_element, matrix_element_configuration);

                    if (parameters.exists("override_parameters")) {
                        const ParameterSet& matrix_element_params = parameters.get<ParameterSet>("override_parameters");
                        auto p = m_ME->getParameters();

                        for (const auto& name: matrix_element_params.getNames()) {
                            double value = matrix_element_params.get<double>(name);
                            p->setParameter(name, value);
                        }

                        p->cacheParameters();
                        p->cacheCouplings();
                    }

                    // PDF, if asked
                    if (use_pdf) {
                        // Silence LHAPDF
                        LHAPDF::setVerbosity(0);

                        std::string pdf = parameters.get<std::string>("pdf");
                        m_pdf.reset(LHAPDF::mkPDF(pdf, 0));

                        double pdf_scale = parameters.get<double>("pdf_scale");
                        pdf_scale_squared = SQ(pdf_scale);
                    }

                    // Prepare arrays for sorting particles
                    for (size_t i = 0; i < m_particles_ids.size(); i++) {
                        indexing.push_back(m_particles_ids[i].me_index - 1);
                    }

                    // Pre-allocate memory for the finalState array
                    finalState.resize(m_particles_ids.size());

                    // Sort the array taking into account the indexing in the configuration
                    std::vector<int64_t> suite(indexing.size());
                    std::iota(suite.begin(), suite.end(), 0);

                    permutations = get_permutations(suite, indexing);
                };

                virtual void beginIntegration() {
                    // Don't assume the non-zero helicities will be the same for each event
                    // In principle they are, but this protects against buggy calls to the ME (e.g. returning NaN or inf)
                    m_ME->resetHelicities();
                    counter = 0 ;
                    std::string out_name = "ME_";
                    static const char alphanum[] =
                        "0123456789"
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        "abcdefghijklmnopqrstuvwxyz";
                    LOG(info)<<" select "<<alphanum;
                    char s[7];
                    for (int i = 0; i < 6; ++i) {
                        s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
                    }
                    s[6] = '\0';
                    std::string s2 = s;
                    out_name += s2 + ".root";
                    if (save_ME){
                        if(gSystem->AccessPathName(out_name.c_str())){
                            out_file = new TFile(out_name.c_str(),"CREATE","MatrixElement");
                            out_tree = new TTree("tree","tree");

                            out_tree->Branch("init1_p4", &init1);
                            out_tree->Branch("init2_p4", &init2);
                            out_tree->Branch("positron_p4", &positron);
                            out_tree->Branch("neutrino_p4", &neutrino);
                            out_tree->Branch("bjet_p4", &bjet);
                            out_tree->Branch("electron_p4", &electron);
                            out_tree->Branch("antineutrino_p4", &antineutrino);
                            out_tree->Branch("antibjet_p4", &antibjet);
                            out_tree->Branch("MEPdf", &MEPdf);
                        }
                        else{
                            out_file = TFile::Open(out_name.c_str(),"UPDATE");
                            out_tree = (TTree*)out_file->Get("tree");
                            out_tree->SetBranchAddress("init1_p4", &init1_ptr);
                            out_tree->SetBranchAddress("init2_p4", &init2_ptr);
                            out_tree->SetBranchAddress("positron_p4", &positron_ptr);
                            out_tree->SetBranchAddress("neutrino_p4", &neutrino_ptr);
                            out_tree->SetBranchAddress("bjet_p4", &bjet_ptr);
                            out_tree->SetBranchAddress("electron_p4", &electron_ptr);
                            out_tree->SetBranchAddress("antineutrino_p4", &antineutrino_ptr);
                            out_tree->SetBranchAddress("antibjet_p4", &antibjet_ptr);
                            out_tree->SetBranchAddress("MEPdf", &MEPdf_ptr);
                        }
                    }
                }


                virtual void endIntegration() {
                    LOG(info)<<"Integration is done";
                    out_file->Write("", TObject::kOverwrite);
                    out_file->Close();
                }

                virtual Status work() override {
                    if (counter>=save_max && save_max!=-1) return Status::OK;


                    static std::vector<LorentzVector> empty_vector;

                    *m_integrand = 0;
                    const std::vector<LorentzVector>& partons = *m_partons;

                    LorentzVectorRefCollection particles;
                    for (const auto& p: m_particles)
                        particles.push_back(std::ref(*p));

                    for (size_t i = 0; i < m_particles_ids.size(); i++) {
                        finalState[i] = std::make_pair(m_particles_ids[i].pdg_id, toVector(particles[i].get()));
                    }

                    // Sort the array taking into account the indexing in the configuration
                    apply_permutations(finalState, permutations);

                    std::pair<std::vector<double>, std::vector<double>> initialState { toVector(partons[0]),
                        toVector(partons[1]) };

                    auto result = m_ME->compute(initialState, finalState);

                    LOG(debug)<<"Inputs";       
                    LOG(debug)<<"Initial states";
                    for (auto const & is: initialState.first){
                        LOG(debug)<<"\tFirst  "<<is;
                    }
                    for (auto const & is: initialState.second){
                        LOG(debug)<<"\tSecond "<<is;
                    }
                    LOG(debug)<<"Final states";
                    for (auto const & fs: finalState){
                        LOG(debug)<<fs.first;
                        for (auto const & f: fs.second)
                            LOG(debug)<<"  "<<f;
                    }
                    for (const auto& me: result) 
                        LOG(debug)<<"Result "<<me.first.first<<" "<<me.first.second<<" "<<me.second;



                    double x1 = std::abs(partons[0].Pz() / (sqrt_s / 2.));
                    double x2 = std::abs(partons[1].Pz() / (sqrt_s / 2.));

                    // Compute flux factor 1/(2*x1*x2*s)
                    double phaseSpaceIn = 1. / (2. * x1 * x2 * SQ(sqrt_s));

                    double integrand = phaseSpaceIn;
                    for (const auto& jacobian: m_jacobians) {
                        integrand *= (*jacobian);
                    }

                    // PDF
                    double final_integrand = 0;
                    if (!use_NN){
                        for (const auto& me: result) {
                            double pdf1 = use_pdf ? m_pdf->xfxQ2(me.first.first, x1, pdf_scale_squared) / x1 : 1;
                            double pdf2 = use_pdf ? m_pdf->xfxQ2(me.first.second, x2, pdf_scale_squared) / x2 : 1;

                            final_integrand += me.second * pdf1 * pdf2;
                        } 
                    }
                    else{
                        std::ifstream input(json_file);
                        lwt::LightweightGraph graph(lwt::parse_json_graph(input));
                        // Evaluate                                                                                                                                                                                         
                        std::map<std::string, std::map<std::string, double> > inputs;

                        inputs["node_0"] = {
                            {"init1_p4.E()", initialState.first[0]}, 
                            {"init2_p4.E()", initialState.second[0]}, 
                            {"positron_p4.Px()", finalState[0].second[1]}, 
                            {"positron_p4.Py()", finalState[0].second[2]}, 
                            {"positron_p4.Pz()", finalState[0].second[3]}, 
                            {"positron_p4.E()", finalState[0].second[0]}, 
                            {"neutrino_p4.Px()", finalState[1].second[1]}, 
                            {"neutrino_p4.Py()", finalState[1].second[2]}, 
                            {"neutrino_p4.Pz()", finalState[1].second[3]}, 
                            {"neutrino_p4.E()", finalState[1].second[0]}, 
                            {"bjet_p4.Px()", finalState[2].second[1]}, 
                            {"bjet_p4.Py()", finalState[2].second[2]}, 
                            {"bjet_p4.Pz()", finalState[2].second[3]}, 
                            {"bjet_p4.E()", finalState[2].second[0]}, 
                            {"electron_p4.Px()", finalState[3].second[1]}, 
                            {"electron_p4.Py()", finalState[3].second[2]}, 
                            {"electron_p4.Pz()", finalState[3].second[3]}, 
                            {"electron_p4.E()", finalState[3].second[0]}, 
                            {"antineutrino_p4.Px()", finalState[4].second[1]}, 
                            {"antineutrino_p4.Py()", finalState[4].second[2]}, 
                            {"antineutrino_p4.Pz()", finalState[4].second[3]}, 
                            {"antineutrino_p4.E()", finalState[4].second[0]}, 
                            {"antibjet_p4.Px()", finalState[5].second[1]}, 
                            {"antibjet_p4.Py()", finalState[5].second[2]}, 
                            {"antibjet_p4.Pz()", finalState[5].second[3]}, 
                            {"antibjet_p4.E()", finalState[5].second[0]}, 
                        };
                        std::map<std::string, double> outputs = graph.compute(inputs);
                        final_integrand = pow(10,-outputs["-log10(MEPdf)"]);
                    }
                    // Save Integral to ROOT

                    if (save_ME){
                        init1.SetPxPyPzE(initialState.first[1],initialState.first[2],initialState.first[3],initialState.first[0]);
                        init2.SetPxPyPzE(initialState.second[1],initialState.second[2],initialState.second[3],initialState.second[0]);
                        positron.SetPxPyPzE(finalState[0].second[1],finalState[0].second[2],finalState[0].second[3],finalState[0].second[0]);
                        neutrino.SetPxPyPzE(finalState[1].second[1],finalState[1].second[2],finalState[1].second[3],finalState[1].second[0]);
                        bjet.SetPxPyPzE(finalState[2].second[1],finalState[2].second[2],finalState[2].second[3],finalState[2].second[0]);
                        electron.SetPxPyPzE(finalState[3].second[1],finalState[3].second[2],finalState[3].second[3],finalState[3].second[0]);
                        antineutrino.SetPxPyPzE(finalState[4].second[1],finalState[4].second[2],finalState[4].second[3],finalState[4].second[0]);
                        antibjet.SetPxPyPzE(finalState[5].second[1],finalState[5].second[2],finalState[5].second[3],finalState[5].second[0]);
                        MEPdf = final_integrand;
                        out_tree->Fill();
                    }



                    LOG(debug)<<"Final integrand "<<final_integrand; 
                    LOG(debug)<<"-----------------------------------";  

                    final_integrand *= integrand;
                    *m_integrand = final_integrand;
                    counter ++;

                    return Status::OK;
                }

                private:
                double sqrt_s;
                bool use_pdf;
                bool save_ME;
                int save_max;
                bool use_NN;
                std::string json_file;
                int counter;
                double pdf_scale_squared = 0;
                std::shared_ptr<momemta::MatrixElement> m_ME;
                std::shared_ptr<LHAPDF::PDF> m_pdf;

                std::vector<int64_t> indexing;
                std::vector<size_t> permutations;
                std::vector<std::pair<int, std::vector<double>>> finalState;

                // Inputs
                Value<std::vector<LorentzVector>> m_partons;

                std::vector<Value<LorentzVector>> m_particles;
                std::vector<ParticleId> m_particles_ids;

                std::vector<Value<double>> m_jacobians;

                // Outputs
                std::shared_ptr<double> m_integrand = produce<double>("output");

                // ROOT saves 
                TLorentzVector init1;            
                TLorentzVector init2;            
                TLorentzVector positron;            
                TLorentzVector neutrino;            
                TLorentzVector bjet;            
                TLorentzVector electron;            
                TLorentzVector antineutrino;            
                TLorentzVector antibjet;            
                TLorentzVector *init1_ptr = &init1;
                TLorentzVector *init2_ptr = &init2;
                TLorentzVector *positron_ptr = &positron;
                TLorentzVector *neutrino_ptr = &neutrino;
                TLorentzVector *bjet_ptr = &bjet;
                TLorentzVector *electron_ptr = &electron;
                TLorentzVector *antineutrino_ptr = &antineutrino;
                TLorentzVector *antibjet_ptr = &antibjet;
                double MEPdf;
                double *MEPdf_ptr = &MEPdf;


                TFile *out_file;
                TTree *out_tree;

            };

REGISTER_MODULE(MatrixElement)
    .Input("initialState")
    .OptionalInputs("jacobians")
    .Inputs("particles/inputs")
    .Output("output")
    .GlobalAttr("energy:double")
    .Attr("matrix_element:string")
    .Attr("matrix_element_parameters:pset")
    .OptionalAttr("override_parameters:pset")
    .Attr("particles:pset")
    .Attr("use_pdf:bool=true")
    .Attr("save_ME:bool=false")
    .Attr("save_max:int=-1")
    .Attr("use_NN:bool=false")
    .Attr("json_file:string")
    .OptionalAttr("pdf:string")
    .OptionalAttr("pdf_scale:double");
