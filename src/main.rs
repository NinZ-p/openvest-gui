use anyhow::{anyhow, Result};
use ndarray::{Array2, ArrayD, Ix2, Ix3};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
    environment::Environment,
};
use std::borrow::Cow;
use tokenizers::Tokenizer;
use std::sync::Arc;
use std::rc::Rc;
use std::cell::RefCell;

use veclite_db::VecLite;
use std::fs;
use std::collections::HashMap;
use itertools::Itertools;

// Embed the model and its external weights into the executable.
const MODEL_BYTES: &[u8] =
    include_bytes!("../models/model_q4.onnx");

const MODEL_DATA_BYTES: &[u8] =
    include_bytes!("../models/model_q4.onnx_data");

const TOKENIZER_BYTES: &[u8] =
    include_bytes!("../models/tokenizer.json");

const MEDIA_FEATURE_SIZE: usize = 512;

pub struct EmbeddingGemma {
    session: Session,
    tokenizer: Tokenizer,
}

impl EmbeddingGemma { // AI generated class. Review later
    pub fn new() -> Result<Self> {
        let env = ort::init()
        .with_name("embedding_gemma")
        .build()?;

        let mut builder = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| anyhow!("Optimization error: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow!("Thread configuration error: {e}"))?
            .with_external_initializer_file_in_memory(
                "model_q4.onnx_data",
                Arc::<[u8]>::from(MODEL_DATA_BYTES),
            )
            .map_err(|e| anyhow!("External weights error: {e}"))?;

        let session = builder
            .commit_from_memory(MODEL_BYTES)
            .map_err(|e| anyhow!("Model loading error: {e}"))?;

        let tokenizer = Tokenizer::from_bytes(TOKENIZER_BYTES)
            .map_err(|e| anyhow!("Tokenizer loading error: {e}"))?;

        Ok(Self {
            session,
            tokenizer,
        })
    }

    /// Embeds a search query.
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        let text = format!(
            "task: search result | query: {}",
            text
        );

        self.embed_internal(&text)
    }

    /// Embeds a document to be indexed.
    pub fn embed_document(&mut self, text: &str) -> Result<Vec<f32>> {
        let text = format!(
            "title: none | text: {}",
            text
        );

        self.embed_internal(&text)
    }

    /// Embeds text without automatically adding a task prefix.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed_internal(text)
    }

    fn embed_internal(&mut self, text: &str) -> Result<Vec<f32>> {
        // Tokenize.
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization error: {e}"))?;

        let ids: Vec<i64> = encoding
            .get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect();

        let mask_values: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&mask| mask as i64)
            .collect();

        let seq_len = ids.len();

        if seq_len == 0 {
            return Err(anyhow!("Tokenizer returned an empty sequence"));
        }

        let input_ids = Tensor::from_array(
            Array2::from_shape_vec((1, seq_len), ids)?
        )?;

        let attention_mask = Tensor::from_array(
            Array2::from_shape_vec(
                (1, seq_len),
                mask_values.clone(),
            )?
        )?;

        // Empty media features for text-only inference.
        let image_features = Tensor::from_array(
            Array2::<f32>::zeros((0, MEDIA_FEATURE_SIZE))
        )?;

        let video_features = Tensor::from_array(
            Array2::<f32>::zeros((0, MEDIA_FEATURE_SIZE))
        )?;

        let audio_features = Tensor::from_array(
            Array2::<f32>::zeros((0, MEDIA_FEATURE_SIZE))
        )?;

        // Run ONNX inference.
        let outputs = self.session.run(ort::inputs! {
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "image_features" => image_features,
            "video_features" => video_features,
            "audio_features" => audio_features,
        })?;

        if outputs.len() == 0 {
            return Err(anyhow!("Model returned no outputs"));
        }

        // Extract the first output.
        let output = outputs[0]
            .try_extract_array::<f32>()?;

        // Convert output to a vector.
        let mut embedding: Vec<f32> = match output.ndim() {
            // Already pooled: [1, hidden_size]
            2 => {
                let array = output
                    .into_dimensionality::<Ix2>()?;

                array.row(0).to_vec()
            }

            // Token embeddings: [1, sequence_length, hidden_size]
            3 => {
                let array = output
                    .into_dimensionality::<Ix3>()?;

                let hidden_size = array.shape()[2];

                let mut pooled = vec![0.0f32; hidden_size];
                let mut token_count = 0.0f32;

                for token_index in 0..seq_len {
                    if mask_values[token_index] == 0 {
                        continue;
                    }

                    for hidden_index in 0..hidden_size {
                        pooled[hidden_index] +=
                            array[[0, token_index, hidden_index]];
                    }

                    token_count += 1.0;
                }

                if token_count == 0.0 {
                    return Err(anyhow!("No valid tokens to pool"));
                }

                for value in pooled.iter_mut() {
                    *value /= token_count;
                }

                pooled
            }

            ndim => {
                return Err(anyhow!(
                    "Unexpected model output rank: {}",
                    ndim
                ));
            }
        };

        // L2 normalization.
        let norm = embedding
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        if norm > 0.0 {
            for value in embedding.iter_mut() {
                *value /= norm;
            }
        }

        Ok(embedding)
    }
}

use dioxus::prelude::*;
use serde_json::{Value};
use serde::{Deserialize, Serialize};
use markdown_dx::{Markdown, Options};
use include_dir::{include_dir, Dir};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use regex::{Captures,Regex};
use std::sync::LazyLock;
use percent_encoding::percent_decode_str;

const LOGO_FULL_SVG: Asset = asset!("/assets/logo_full.svg");
const LOGO_SVG: Asset = asset!("/assets/logo_red.svg");
const MAIN_CSS: Asset = asset!("/assets/main.css");
const HOME_SVG: Asset = asset!("/assets/home.svg");
const EXER_SVG: Asset = asset!("/assets/exercicios.svg");
const DESE_SVG: Asset = asset!("/assets/desempenho.svg");
const LUPA_SVG: Asset = asset!("/assets/lupa.svg");
const DICA_SVG: Asset = asset!("/assets/dica.svg");
const GABA_SVG: Asset = asset!("/assets/gabarito.svg");

const PROVAS_JSON: &[u8] = include_bytes!("../assets/tstast_monolith.json");

//println!(loaded_json);

#[derive(Clone,Copy,PartialEq)]
enum Tab {
    THome,
    TExercicio,
    TDesempenho,
}

#[derive(Serialize, Deserialize, Clone)]
struct Question {
    id_prova: String,
    numero_exercicio: i32,
    enunciado: String,
    alternativas: Vec<String>,
    gabarito: i32,
    comentarios: Vec<String>,
    materias: Vec<String>,
    conteudos: Vec<String>,
    dica: String
}

const DB_BYTES: &[u8] = include_bytes!("../assets/tstast_db.vec");

fn load_database() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join("rtdb.vec");

    fs::write(&path, DB_BYTES)?;
    Ok(())
}

fn main() {
    load_database();
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let model = use_hook(|| {
        Rc::new(RefCell::new(EmbeddingGemma::new()))
    });

    let provas = use_resource(|| async move {
        match serde_json::from_slice::<Vec<Question>>(PROVAS_JSON) {
            Ok(questions) => {
                println!("Loaded {} questions", questions.len());
                Some(questions)
            }
            Err(err) => {
                println!("JSON parsing error: {err}");
                None
            }
        }
    });

    let mut loaded_questions=use_signal(|| Vec::<usize>::new());
    let mut user_questions=use_signal(|| Vec::<usize>::new());
    let mut active_tab=use_signal(|| Tab::THome);
    let mut user_answer=use_signal(|| -1);

    let mut is_dica_active=use_signal(|| false);

    let mut question_list_idx=use_signal(|| 0usize);

    let lista_materias: Signal<Vec<String>> = use_signal(|| vec![
        "linguagens, códigos e suas tecnologias".to_string(),
        "matemática e suas tecnologias".to_string(),
        "ciências da natureza e suas tecnologias".to_string(),
        "ciências humanas e suas tecnologias".to_string(),
    ]);

    let dict_conteudos_materias: Signal<HashMap<String,String>> = use_signal(|| {[
        // ===== Linguagens | competências de área =====
        ("linguagens | competência de área 1", "linguagens, códigos e suas tecnologias"),
        ("linguagens | competência de área 2", "linguagens, códigos e suas tecnologias"),
        ("linguagens | competência de área 3", "linguagens, códigos e suas tecnologias"),
        ("linguagens | competência de área 4", "linguagens, códigos e suas tecnologias"),
        ("linguagens | competência de área 5", "linguagens, códigos e suas tecnologias"),
        ("linguagens | competência de área 6", "linguagens, códigos e suas tecnologias"),
        ("linguagens | competência de área 7", "linguagens, códigos e suas tecnologias"),
        ("linguagens | competência de área 8", "linguagens, códigos e suas tecnologias"),
        ("linguagens | competência de área 9", "linguagens, códigos e suas tecnologias"),

        // ===== Matemática | competências de área =====
        ("matemática | competência de área 1", "matemática e suas tecnologias"),
        ("matemática | competência de área 2", "matemática e suas tecnologias"),
        ("matemática | competência de área 3", "matemática e suas tecnologias"),
        ("matemática | competência de área 4", "matemática e suas tecnologias"),
        ("matemática | competência de área 5", "matemática e suas tecnologias"),
        ("matemática | competência de área 6", "matemática e suas tecnologias"),

        // ===== Ciências da Natureza | competências de área =====
        ("ciências da natureza | competência de área 1", "ciências da natureza e suas tecnologias"),
        ("ciências da natureza | competência de área 2", "ciências da natureza e suas tecnologias"),
        ("ciências da natureza | competência de área 3", "ciências da natureza e suas tecnologias"),
        ("ciências da natureza | competência de área 4", "ciências da natureza e suas tecnologias"),
        ("ciências da natureza | competência de área 5", "ciências da natureza e suas tecnologias"),
        ("ciências da natureza | competência de área 6", "ciências da natureza e suas tecnologias"),
        ("ciências da natureza | competência de área 7", "ciências da natureza e suas tecnologias"),

        // ===== Ciências Humanas | competências de área =====
        ("ciências humanas | competência de área 1", "ciências humanas e suas tecnologias"),
        ("ciências humanas | competência de área 2", "ciências humanas e suas tecnologias"),
        ("ciências humanas | competência de área 3", "ciências humanas e suas tecnologias"),
        ("ciências humanas | competência de área 4", "ciências humanas e suas tecnologias"),
        ("ciências humanas | competência de área 5", "ciências humanas e suas tecnologias"),
        ("ciências humanas | competência de área 6", "ciências humanas e suas tecnologias"),

        // ===== Objetos de conhecimento: Linguagens =====
        ("estudo do texto", "linguagens, códigos e suas tecnologias"),
        ("práticas corporais", "linguagens, códigos e suas tecnologias"),
        ("textos artísticos", "linguagens, códigos e suas tecnologias"),
        ("texto literário", "linguagens, códigos e suas tecnologias"),
        ("aspectos linguísticos em diferentes textos", "linguagens, códigos e suas tecnologias"),
        ("texto argumentativo", "linguagens, códigos e suas tecnologias"),
        ("usos da língua portuguesa", "linguagens, códigos e suas tecnologias"),
        ("gêneros digitais", "linguagens, códigos e suas tecnologias"),

        // ===== Objetos de conhecimento: Matemática =====
        ("conhecimentos numéricos", "matemática e suas tecnologias"),
        ("conhecimentos geométricos", "matemática e suas tecnologias"),
        ("estatística e probabilidade", "matemática e suas tecnologias"),
        ("conhecimentos algébricos", "matemática e suas tecnologias"),
        ("conhecimentos algébricos/geométricos", "matemática e suas tecnologias"),

        // ===== Objetos de conhecimento: Ciências da Natureza (Física) =====
        ("conhecimentos básicos", "ciências da natureza e suas tecnologias"),
        ("mecânica, movimento e equilíbrio", "ciências da natureza e suas tecnologias"),
        ("energia, trabalho e potência", "ciências da natureza e suas tecnologias"),
        ("mecânica celeste", "ciências da natureza e suas tecnologias"),
        ("fenômenos elétricos e magnéticos", "ciências da natureza e suas tecnologias"),
        ("oscilações, ondas, óptica e radiação", "ciências da natureza e suas tecnologias"),
        ("calor e fenômenos térmicos", "ciências da natureza e suas tecnologias"),
        
        // ===== Objetos de conhecimento: Ciências da Natureza (Química) =====
        ("transformações químicas", "ciências da natureza e suas tecnologias"),
        ("representação das transformações", "ciências da natureza e suas tecnologias"),
        ("materiais e propriedades", "ciências da natureza e suas tecnologias"),
        ("estudo da água", "ciências da natureza e suas tecnologias"),
        ("transformações químicas e energia", "ciências da natureza e suas tecnologias"),
        ("dinâmica e equilíbrio químico", "ciências da natureza e suas tecnologias"),
        ("compostos de carbono", "ciências da natureza e suas tecnologias"),
        ("aplicações tecnológicas e socioambientais", "ciências da natureza e suas tecnologias"),
        ("energias químicas no cotidiano", "ciências da natureza e suas tecnologias"),

        // ===== Objetos de conhecimento: Ciências da Natureza (Biologia) =====
        ("moléculas, células e tecidos", "ciências da natureza e suas tecnologias"),
        ("hereditariedade e genética", "ciências da natureza e suas tecnologias"),
        ("identidade dos seres vivos", "ciências da natureza e suas tecnologias"),
        ("ecologia e ciências ambientais", "ciências da natureza e suas tecnologias"),
        ("origem e evolução da vida", "ciências da natureza e suas tecnologias"),
        ("qualidade de vida das populações humanas", "ciências da natureza e suas tecnologias"),

        // ===== Objetos de conhecimento: Ciências Humanas =====
        ("diversidade cultural, conflitos e vida em sociedade", "ciências humanas e suas tecnologias"),
        ("organização social, pensamento político e estado", "ciências humanas e suas tecnologias"),
        ("estruturas produtivas", "ciências humanas e suas tecnologias"),
        ("espaços agrários", "ciências humanas e suas tecnologias"),
        ("meio ambiente e sociedade", "ciências humanas e suas tecnologias"),
        ("representação espacial", "ciências humanas e suas tecnologias"),
    ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    });

    let lista_conteudos: Signal<Vec<String>> = use_signal(||
        [
            // ===== Linguagens | competências de área =====
            "linguagens | competência de área 1",
            "linguagens | competência de área 2",
            "linguagens | competência de área 3",
            "linguagens | competência de área 4",
            "linguagens | competência de área 5",
            "linguagens | competência de área 6",
            "linguagens | competência de área 7",
            "linguagens | competência de área 8",
            "linguagens | competência de área 9",

            // ===== Matemática | competências de área =====
            "matemática | competência de área 1",
            "matemática | competência de área 2",
            "matemática | competência de área 3",
            "matemática | competência de área 4",
            "matemática | competência de área 5",
            "matemática | competência de área 6",

            // ===== Ciências da Natureza | competências de área =====
            "ciências da natureza | competência de área 1",
            "ciências da natureza | competência de área 2",
            "ciências da natureza | competência de área 3",
            "ciências da natureza | competência de área 4",
            "ciências da natureza | competência de área 5",
            "ciências da natureza | competência de área 6",
            "ciências da natureza | competência de área 7",

            // ===== Ciências Humanas | competências de área =====
            "ciências humanas | competência de área 1",
            "ciências humanas | competência de área 2",
            "ciências humanas | competência de área 3",
            "ciências humanas | competência de área 4",
            "ciências humanas | competência de área 5",
            "ciências humanas | competência de área 6",

            // ===== Objetos de conhecimento: Linguagens =====
            "estudo do texto",
            "práticas corporais",
            "textos artísticos",
            "texto literário",
            "aspectos linguísticos em diferentes textos",
            "texto argumentativo",
            "usos da língua portuguesa",
            "gêneros digitais",

            // ===== Objetos de conhecimento: Matemática =====
            "conhecimentos numéricos",
            "conhecimentos geométricos",
            "estatística e probabilidade",
            "conhecimentos algébricos",
            "conhecimentos algébricos/geométricos",

            // ===== Objetos de conhecimento: Física =====
            "conhecimentos básicos",
            "mecânica, movimento e equilíbrio",
            "energia, trabalho e potência",
            "mecânica celeste",
            "fenômenos elétricos e magnéticos",
            "oscilações, ondas, óptica e radiação",
            "calor e fenômenos térmicos",

            // ===== Objetos de conhecimento: Química =====
            "transformações químicas",
            "representação das transformações",
            "materiais e propriedades",
            "estudo da água",
            "transformações químicas e energia",
            "dinâmica e equilíbrio químico",
            "compostos de carbono",
            "aplicações tecnológicas e socioambientais",
            "energias químicas no cotidiano",

            // ===== Objetos de conhecimento: Biologia =====
            "moléculas, células e tecidos",
            "hereditariedade e genética",
            "identidade dos seres vivos",
            "ecologia e ciências ambientais",
            "origem e evolução da vida",
            "qualidade de vida das populações humanas",

            // ===== Objetos de conhecimento: Ciências Humanas =====
            "diversidade cultural, conflitos e vida em sociedade",
            "organização social, pensamento político e estado",
            "estruturas produtivas",
            "espaços agrários",
            "meio ambiente e sociedade",
            "representação espacial",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    );

    let mut perf_mat_cont:Signal<HashMap<String,(i32,i32)>>=use_signal(|| HashMap::new());
    
    let perf_mat_cont: Signal<HashMap<String,(i32,i32)>> = use_signal(|| {
        let mut m = HashMap::new();
        for mat in lista_materias.read().iter() { m.insert(mat.clone(), (0,0)); }
        for cont in lista_conteudos.read().iter() { m.insert(cont.clone(), (0,0)); }
        m
    });

    rsx! {
        document::Link { rel: "icon", href: LOGO_SVG},
        document::Link { rel: "stylesheet", href: MAIN_CSS },
        document::Title { "Openvest" },
        TopBarMenu { active_tab },
        match *active_tab.read() {
            Tab::THome => {Home(HomeProps{provas:provas,model: SharedEmbeddingModel(model.clone()),loaded_questions:loaded_questions,user_questions:user_questions})},
            Tab::TExercicio => {Exercicio(ExercicioProps{provas:provas,user_answer:user_answer,user_questions:user_questions,question_list_idx:question_list_idx,perf_mat_cont:perf_mat_cont,is_dica_active:is_dica_active})},
            Tab::TDesempenho => {Desempenho(DesempenhoProps{lista_materias:lista_materias,dict_conteudos_materias:dict_conteudos_materias,perf_mat_cont:perf_mat_cont})},
            _ => rsx! {div{"Error"}}
        }
    }
}

static IMAGES: Dir<'_> =
    include_dir!("$CARGO_MANIFEST_DIR/assets/pimgs");

// Detecta referências Markdown de imagens.
static IMAGE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"!\[([^\]]*)\]\((<[^>]+>|[^\s)]+)(?:\s+("[^"]*"|'[^']*'))?\)"#
    ).unwrap()
});

// Localiza uma imagem no diretório incorporado,
// mesmo quando o Markdown contém um caminho absoluto.
fn get_image_data_url(raw_path: &str) -> Option<String> {

    // Remove delimitadores Markdown.
    let path = raw_path
        .trim_start_matches('<')
        .trim_end_matches('>');

    // Remove file:// de caminhos absolutos.
    let path = path
        .strip_prefix("file://")
        .unwrap_or(path);

    // Remove query strings e fragments.
    let path = path
        .split(['?', '#'])
        .next()
        .unwrap_or(path);

    // Decodifica caracteres como %20 e normaliza separadores.
    let decoded = percent_decode_str(path)
        .decode_utf8_lossy()
        .replace('\\', "/");

    // Divide o caminho em componentes.
    let parts: Vec<&str> = decoded
        .split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .collect();

    // Tenta localizar o arquivo usando todos os sufixos
    // do caminho. Isso permite encontrar imagens mesmo
    // quando o Markdown contém caminhos absolutos.
    for start in 0..parts.len() {

        let relative_path = parts[start..].join("/");

        if let Some(file) = IMAGES.get_file(&relative_path) {

            let mime = mime_guess::from_path(&relative_path)
                .first_or_octet_stream();

            let encoded = STANDARD.encode(file.contents());

            return Some(format!(
                "data:{};base64,{}",
                mime,
                encoded
            ));
        }
    }

    None
}

// Substitui somente o caminho da imagem,
// preservando a sintaxe Markdown original.
fn replace_images(markdown: &str) -> String {

    IMAGE_REGEX
        .replace_all(markdown, |caps: &Captures| {

            let alt = &caps[1];

            let raw_path = &caps[2];

            match get_image_data_url(raw_path) {

                Some(data_url) => {

                    // Continua sendo Markdown!
                    format!(
                        "![{}]({})",
                        alt,
                        data_url
                    )
                }

                None => {

                    eprintln!(
                        "Imagem não encontrada: {}",
                        raw_path
                    );

                    // Mantém a referência original.
                    caps[0].to_string()
                }
            }
        })
        .into_owned()
}

#[component]
pub fn MarkdownView(content: String) -> Element {

    let custom_options = Options::builder()
        .disable_default_theme(true)
        .build();

    let processed_content = replace_images(&content);

    rsx! {
        Markdown {
            src: processed_content,
            options: custom_options,
        }
    }
}

#[component]
pub fn TopBarMenu(mut active_tab: Signal<Tab>) -> Element {
    rsx! {
        div {
            id: "TopBarMenu",
            div{
                onclick: move |_| active_tab.set(Tab::THome),
                id: "Menuitem",
                div{
                    id: "Icon",
                    img {src:HOME_SVG}
                },
                "Menu",
            }
            div{
                onclick: move |_| active_tab.set(Tab::TExercicio),
                id: "Menuitem",
                div{
                    id: "Icon",
                    img {src:EXER_SVG}
                },
                "Exercícios",
            }
            div{
                onclick: move |_| active_tab.set(Tab::TDesempenho),
                id: "Menuitem",
                div{
                    id: "Icon",
                    img {src:DESE_SVG}
                },
                "Desempenho",
            }
        }
    }
}

#[derive(Clone)]
pub struct SharedEmbeddingModel(
    Rc<RefCell<Result<EmbeddingGemma, anyhow::Error>>>
);

impl PartialEq for SharedEmbeddingModel {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
fn ov_search(query: &String,model: &SharedEmbeddingModel,provas:Resource<Option<Vec<Question>>>) -> anyhow::Result<Vec<usize>> {
    let mut model = model.0.borrow_mut();

    let model = model
        .as_mut()
        .map_err(|e| anyhow!("Model initialization failed: {e}"))?;
        
    let mut func_result:Vec<usize>=vec!();

    // Generate an embedding.
    let embedding = model.embed(query)?;

    let mut db = VecLite::open(std::env::temp_dir().join("rtdb.vec")).unwrap();

    let results = db
        .build_search(&embedding)
        .top_k(16)
        .execute()?;

    for (i, result) in results.iter().enumerate() {
        let id_num:usize=result.id
            .trim()
            .parse()
            .expect("Invalid number");
        func_result.push(id_num)
    }



    drop(db);
    Ok(func_result)
}

#[component]
pub fn Home(provas:Resource<Option<Vec<Question>>>,model: SharedEmbeddingModel,loaded_questions:Signal<Vec<usize>>, user_questions:Signal<Vec<usize>>) -> Element {
    let mut search_query=use_signal(String::new);
    let mut searched_query=use_signal(String::new);
    
    let model_click = model.clone();
    let model_enter = model.clone();

    rsx! {
        div {
            id:"MenuTitle",
            "Comece a estudar"
        }

        div {
            id:"SearchBar",
            div {
                id:"SearchBarIcon",
                onclick: move |_| {
                    searched_query.set(search_query.read().to_string());
                    loaded_questions.set(
                        match (ov_search(&searched_query.read(),&model_click,provas)) {
                            Ok(value) => value,
                            Err(err) => {
                                println!("Error: {err}");
                                return;
                            }
                        }
                    );
                },
                div {
                    img {src:LUPA_SVG}
                }
            },
            input {
                id:"SearchText",
                r#type:"text",
                placeholder:"Clique e começe a pesquisar",
                value:"{search_query}",
                oninput: move |event| search_query.set(event.value()),
                onkeydown: move |evt| {
                    if evt.key() == Key::Enter {
                        searched_query.set(search_query.read().to_string());
                        loaded_questions.set(
                            match (ov_search(&searched_query.read(),&model_enter,provas)) {
                                Ok(value) => value,
                                Err(err) => {
                                    println!("Error: {err}");
                                    return;
                                }
                            }
                        );
                        
                        spawn(async move {
                            let _ = document::eval(r#"
                                document.activeElement?.blur();
                            "#).await;
                        });
                    }
                },

            }
        }

        div {
            id:"SearchResultBox",
            div{
                id: "SearchedQuestion",
                div {
                    id: "GabaritoText",
                    b {"{loaded_questions.read().len()} Resultados"}
                }
                if (!loaded_questions.read().iter().all(|x| user_questions.read().contains(x))) {
                    div {
                        id:"AddAllButton",
                        onclick: move |_| {
                            for q in &*loaded_questions.read() {
                                if (!user_questions.read().contains(&*q)) {
                                    user_questions.write().push(*q);
                                }
                            }
                        },
                        "Adicionar todos"
                    }
                }
                else {
                    div {
                        id:"AddAllButtonDisabled",
                        "Adicionar todos"
                    }
                }
            }
            if let Some(Some(json)) = &*provas.read() {
                for (i,qstid) in loaded_questions.read().clone().into_iter().enumerate() {
                    div{
                        id: "SearchedQuestion",
                        div{
                            id: "GabaritoText",
                            "{json[qstid].id_prova} - {json[qstid].numero_exercicio}"
                        },
                        if (user_questions.read().contains(&qstid)) {
                            div{
                                id: "Botao_Pressed",
                                "-"
                            }
                        }
                        else {
                            div{
                                id: "Botao",
                                onclick: move |_| {
                                    user_questions.write().push(loaded_questions.read()[i])
                                },
                                "+"
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub fn Exercicio(provas:Resource<Option<Vec<Question>>>,user_answer:Signal<i32>,user_questions:Signal<Vec<usize>>,question_list_idx:Signal<usize>,perf_mat_cont:Signal<HashMap<String,(i32,i32)>>,is_dica_active:Signal<bool>) -> Element {    
    let mut current_id:String="".to_string();
    let mut current_ex_num:String="".to_string();
    let mut current_top_text:String="".to_string();
    let mut current_enunciado:String="".to_string();
    let mut current_alternativas:Vec<String>=vec!(
        "".to_string(),
        "".to_string(),
        "".to_string(),
        "".to_string(),
        "".to_string()
    );
    let mut current_gabarito:i32=-1;
    let mut current_explanation:Vec<String>=vec!(
        "".to_string(),
        "".to_string(),
        "".to_string(),
        "".to_string(),
        "".to_string()
    );
    let mut current_explanation_str:String="".to_string();

    let mut current_materias:Vec<String>=vec![];
    let mut current_conteudos:Vec<String>=vec![];

    let mut current_dica:String="".to_string();
    
    if (*question_list_idx.read()<user_questions.read().len()) {
        if let Some(Some(json)) = &*provas.read() {
            let mut current_question:&Question=&json[user_questions.read()[*question_list_idx.read()]];
            current_id=(*current_question.id_prova).to_string();
            current_ex_num=(current_question.numero_exercicio).to_string();
            current_top_text=format!("{} - {}",current_id,current_ex_num);
            current_enunciado=(*current_question.enunciado).to_string();
            current_alternativas[0]=(*current_question.alternativas)[0].to_string();
            current_alternativas[1]=(*current_question.alternativas)[1].to_string();
            current_alternativas[2]=(*current_question.alternativas)[2].to_string();
            current_alternativas[3]=(*current_question.alternativas)[3].to_string();
            current_alternativas[4]=(*current_question.alternativas)[4].to_string();
            current_gabarito=((*current_question).gabarito);
            current_explanation[0]=(*current_question.comentarios)[0].to_string();
            current_explanation[1]=(*current_question.comentarios)[1].to_string();
            current_explanation[2]=(*current_question.comentarios)[2].to_string();
            current_explanation[3]=(*current_question.comentarios)[3].to_string();
            current_explanation[4]=(*current_question.comentarios)[4].to_string();
            current_materias=(*current_question.materias).to_vec();
            current_conteudos=(*current_question.conteudos).to_vec();
            current_explanation_str=format!(
                "- **A** - {}  \n- **B** - {}  \n- **C** - {}  \n- **D** - {}  \n- **E** - {}  \n",
                current_explanation[0],
                current_explanation[1],
                current_explanation[2],
                current_explanation[3],
                current_explanation[4]
            );
            current_dica=(*current_question.dica).to_string();
        }
        let rc_materias = Rc::new(current_materias);
        let rc_conteudos = Rc::new(current_conteudos);
        rsx! {
            div {
                id: "Exercicio",
                div{
                    id: "ExercicioBox",
                    div{
                        strong {"{current_top_text}"}
                    }
                    div{
                        id: "QuestBox",
                        MarkdownView {content: current_enunciado}
                    }
                    for alt_id in 0i32..5i32 {
                        div{
                            id: 
                                if (*user_answer.read()==-1) {
                                    "QuestContainer"
                                } else if (current_gabarito==-1 || current_gabarito==alt_id) {
                                    "QuestContainerRight"
                                } else {
                                    "QuestContainerWrong"
                                },
                            onclick: {
                                let rcc_materias = Rc::clone(&rc_materias);
                                let rcc_conteudos = Rc::clone(&rc_conteudos);

                                move |_| {
                                    if (*user_answer.read()==-1) { 
                                        for mat in &*rcc_materias {
                                            if perf_mat_cont.read().contains_key(&mat.clone()) {
                                                if (current_gabarito==-1 || current_gabarito==alt_id) {perf_mat_cont.write().entry(mat.clone().to_string()).or_insert((0,0)).0 += 1;}
                                                else{perf_mat_cont.write().entry(mat.clone().to_string()).or_insert((0,0));};
                                                perf_mat_cont.write().entry(mat.clone().to_string()).or_insert((0,0)).1 += 1;
    //                                            println!("{}",mat);
                                            }
                                        }
                                        for cont in &*rcc_conteudos {
                                            if perf_mat_cont.read().contains_key(&cont.clone()) {
                                                if (current_gabarito==-1 || current_gabarito==alt_id) {perf_mat_cont.write().entry(cont.clone().to_string()).or_insert((0,0)).0 += 1;}
                                                else {perf_mat_cont.write().entry(cont.clone().to_string()).or_insert((0,0));};
                                                perf_mat_cont.write().entry(cont.clone().to_string()).or_insert((0,0)).1 += 1;
    //                                            println!("{}",cont);
                                            }
                                        }
                                        user_answer.set(alt_id);
                                    }
                                }
                            },
                            div{
                                id: "QuestID",
                                "{vec![\"A\",\"B\",\"C\",\"D\",\"E\"][alt_id as usize]}"
                            }
                            div{
                                id: "QuestAnswer",
                                MarkdownView {content: <std::string::String as Clone>::clone(&current_alternativas[alt_id as usize])}
                            }
                        },
                    },
                    if (!*is_dica_active.read() && *user_answer.read()==-1) {
                        div{
                            id:"QuestContainer",
                            onclick: move |_| is_dica_active.set(true),
                            div{
                                id:"QuestID",
                                div{
                                    id:"Icon",
                                    img{src:DICA_SVG}
                                }
                            }
                            div{
                                id:"QuestAnswer",
                                b{"Dica"}
                            }
                        }
                    }
                    else if (*user_answer.read()==-1) {
                        div {
                            id:"Gabarito",
                            div{
                                id: "DicaBox",
                                div{
                                    id: "Icon",
                                    img {src:DICA_SVG}
                                }
                                b {"Dica"}
                                
                            }
                            div{
                                id:"GabaritoText",
                                "{current_dica}"
                            }
                        }
                    }
                    match *user_answer.read() {
                        -1 => rsx!{},
                        _ => rsx! {
                            div{
                                id:"Gabarito",
                                div{
                                    id: "GabaritoBox",
                                    b {"Gabarito"}
                                    div{
                                        id: "Icon",
                                        img {src:GABA_SVG}
                                    }
                                }
                                div{
                                    id:"GabaritoText",
                                    MarkdownView {content: <std::string::String as Clone>::clone(&current_explanation_str)}
                                }
                            }
                        }
                    }
                }
                div{
                    id: "ProximoExercicio",
                    onclick: move |_| {
                        let current = *question_list_idx.read();
                        user_questions.remove(0);
                        user_answer.set(-1);
                        is_dica_active.set(false);
                    },
                    b {"Próximo exercício"}
                }
            }
        }
    }
    else {
        rsx! {
            div {
                id: "MenuTitle",
                "Adicione exercícios na página anterior!"
            }
        }
    }
}

#[component]
//pub fn Desempenho(materiasEstudadas: &[i32], assuntosEstudadas: &[i32]) -> Element {
pub fn Desempenho(lista_materias:Signal<Vec<String>>,dict_conteudos_materias:Signal<HashMap<String,String>>,perf_mat_cont:Signal<HashMap<String,(i32,i32)>>) -> Element {
    rsx! {
        div{
            id: "DesempenhoLegenda",
            div{
                width: "80%",
                margin: "auto",
                div {
                    id: "OrganizeRowAdaptText",
                    width: "60%",
                    div{
                        id: "OrganizeRow",
                        width: "fit-content",
                        div{
                            id: "DesmpenhoR"
                        }
                        "<50%"
                    }
                    div{
                        id: "OrganizeRow",
                        width: "fit-content",
                        div{
                            id: "DesmpenhoO"
                        }
                        "50%-69%"
                    }
                    div{
                        id: "OrganizeRow",
                        width: "fit-content",
                        div{
                            id: "DesmpenhoB"
                        }
                        "70%-89%"
                    }
                    div{
                        id: "OrganizeRow",
                        width: "fit-content",
                        div{
                            id: "DesmpenhoMB"
                        }
                        "89%-99%"
                    }
                    div{
                        id: "OrganizeRow",
                        width: "fit-content",
                        div{
                            id: "DesmpenhoP"
                        }
                        "100%"
                    }
                }
            }
        },

        div{
            id: "DesempenhoContainer",
            div{
                id:"OrganizeRowFlex",
                div {
                    id: "TextoDesempenho",
                    div{
                        b {"Disciplinas - xx"}
                    }
                    div{
                        "Distribuição por desempenho"
                    }
                    div{
                        id: "OrganizeRow",
                        div{
                            color: "rgb(226, 53, 53)",
                            b {"Ruim"}
                        }
                        div{
                            color: "rgb(231, 151, 32)",
                            b {"Ok"}
                        }
                        div{
                            color: "rgb(27, 143, 1)",
                            b {"Bom"}
                        }
                        div{
                            color: "rgb(26, 71, 37)",
                            b {"Muito Bom"}
                        }
                        div{
                            color: "rgb(115, 50, 190)",
                            b {"Perfeito"}
                        }
                    }
                    div{
                        id: "ProgressBar",
                        div{
                            id: "PBR",
                            width: "20%"
                        } 
                        div{
                            id: "PBO",
                            width: "20%"
                        }        
                        div{
                            id: "PBB",
                            width: "20%"
                        }        
                        div{
                            id: "PBMB",
                            width: "20%"
                        }        
                        div{
                            id: "PBP",
                            width: "20%"
                        }                              
                    }
                }
                div {
                    id: "TextoDesempenho",
                    div{
                        b {"Assuntos - xx"}
                    }
                    div{
                        "Distribuição por desempenho"
                    }
                    div{
                        id: "OrganizeRow",
                        div{
                            color: "rgb(226, 53, 53)",
                            b {"Ruim"}
                        }
                        div{
                            color: "rgb(231, 151, 32)",
                            b {"Ok"}
                        }
                        div{
                            color: "rgb(27, 143, 1)",
                            b {"Bom"}
                        }
                        div{
                            color: "rgb(26, 71, 37)",
                            b {"Muito Bom"}
                        }
                        div{
                            color: "rgb(115, 50, 190)",
                            b {"Perfeito"}
                        }
                    }
                    div{
                        id: "ProgressBar",
                        div{
                            id: "PBR",
                            width: "20%"
                        } 
                        div{
                            id: "PBO",
                            width: "20%"
                        }        
                        div{
                            id: "PBB",
                            width: "20%"
                        }        
                        div{
                            id: "PBMB",
                            width: "20%"
                        }        
                        div{
                            id: "PBP",
                            width: "20%"
                        }                              
                    }
                }
            }
        }

        div{
            width: "80%",
            margin: "auto",
            
            div{
                id:"OrganizeRow",
                padding: "0px 0px 30px 0px",
                div{
                    id: "LargerText",
                    b {"Matéria / Conteúdo"}
                }
                div{
                    id: "LargerText",
                    b {"Taxa de Acerto"}
                }
            }

            for mat in lista_materias.read().clone().into_iter() {
                if (perf_mat_cont
                    .read()
                    .get(&mat)
                    .expect("Content not on content_list")
                    .1
                >0) {
                    div{
                        id:"OrganizeRow",
                        padding: "5px",
                        div{
                            id:"MateriaEBotao",
                            div{
                                id: "LargerText",
                                b {"{mat}"}
                            }
//                            div{
//                                id: "ExpandButton",
//                                "V"
//                            }
                        }

                        div{
                            id:"OrganizeRowNOGAP",
                            padding: "0px",
                            div{
                                background_color: {
                                    let srate={
                                        (100*perf_mat_cont
                                            .read()
                                            .get(&mat)
                                            .expect("Content not on content_list")
                                            .0)
                                        .checked_div(perf_mat_cont
                                            .read()
                                            .get(&mat)
                                            .expect("Content not on content_list")
                                            .1)
                                        .unwrap_or(0)
                                    };
                                    if srate<50 {
                                         "rgb(226, 53, 53)"
                                    }
                                    else if srate<70 {
                                         "rgb(231, 151, 32)"
                                    }
                                    else if srate<90 {
                                         "rgb(42, 215, 30)"
                                    }
                                    else if srate<100 {
                                         "rgb(43, 121, 61)"
                                    }
                                    else {
                                         "rgb(115, 50, 190)"
                                    }
                                },
                                width: {
                                    let srate={
                                        (100*perf_mat_cont
                                            .read()
                                            .get(&mat)
                                            .expect("Content not on content_list")
                                            .0)
                                        .checked_div(perf_mat_cont
                                            .read()
                                            .get(&mat)
                                            .expect("Content not on content_list")
                                            .1)
                                        .unwrap_or(0)
                                    };
                                    format!("{}vw",srate/2)
                                },
                                id: "ProgressBar",
                                height: "1vh"
                            }
                            div{
                                "{perf_mat_cont
                                    .read()
                                    .get(&mat)
                                    .expect(\"Content not on content_list\")
                                    .0
                                }/{perf_mat_cont
                                    .read()
                                    .get(&mat)
                                    .expect(\"Content not on content_list\")
                                    .1
                                }({(100*perf_mat_cont
                                    .read()
                                    .get(&mat)
                                    .expect(\"Content not on content_list\")
                                    .0)
                                .checked_div(perf_mat_cont
                                    .read()
                                    .get(&mat)
                                    .expect(\"Content not on content_list\")
                                    .1)
                                .unwrap_or(0)}%)"
                            }
                        }
                    }
                    for cont in dict_conteudos_materias.read()
                        .iter()
                        .filter(|(_, &ref v)| *v == mat)
                        .map(|(k, _)| k.clone())
                        .sorted()
                        .collect::<Vec<String>>()
                    {
                        if (perf_mat_cont
                            .read()
                            .get(&cont)
                            .expect("Content not on content_list")
                            .1
                        >0) {
                            div{
                                id:"OrganizeRow",
                                style: "margin-left: auto;",
                                width: "98%",
                                padding: "5px 0px 0px 0px",
                                div{
                                    padding: "0px 15px 0px 0px",
                                    "{cont}"
                                }
                                div{
                                    id:"OrganizeRowNOGAP",
                                    padding: "0px",
                                    div{
                                        width: {
                                            let srate={
                                                (100*perf_mat_cont
                                                    .read()
                                                    .get(&mat)
                                                    .expect("Content not on content_list")
                                                    .0)
                                                .checked_div(perf_mat_cont
                                                    .read()
                                                    .get(&mat)
                                                    .expect("Content not on content_list")
                                                    .1)
                                                .unwrap_or(0)
                                            };
                                            format!("{}vw",srate/2)
                                        },
                                        background_color: {
                                            let srate={
                                                (100*perf_mat_cont
                                                    .read()
                                                    .get(&cont)
                                                    .expect("Content not on content_list")
                                                    .0)
                                                .checked_div(perf_mat_cont
                                                    .read()
                                                    .get(&cont)
                                                    .expect("Content not on content_list")
                                                    .1)
                                                .unwrap_or(0)
                                            };
                                            if srate<50 {
                                                 "rgb(226, 53, 53)"
                                            }
                                            else if srate<70 {
                                                 "rgb(231, 151, 32)"
                                            }
                                            else if srate<90 {
                                                 "rgb(42, 215, 30)"
                                            }
                                            else if srate<100 {
                                                 "rgb(43, 121, 61)"
                                            }
                                            else {
                                                 "rgb(115, 50, 190)"
                                            }
                                        },
                                            
                                        id: "ProgressBar",
                                        height: "1vh",
                                    }
                                    div{
                                        "{perf_mat_cont
                                            .read()
                                            .get(&cont)
                                            .expect(\"Content not on content_list\")
                                            .0
                                        }/{perf_mat_cont
                                            .read()
                                            .get(&cont)
                                            .expect(\"Content not on content_list\")
                                            .1
                                        }({(100*perf_mat_cont
                                            .read()
                                            .get(&cont)
                                            .expect(\"Content not on content_list\")
                                            .0)
                                        .checked_div(perf_mat_cont
                                            .read()
                                            .get(&cont)
                                            .expect(\"Content not on content_list\")
                                            .1)
                                        .unwrap_or(0)}%)"
                                    }
                                }
                            }
                        }
                        else {
                            div {}
                        }
                    }
                }
                else {
                    div {}
                }
            }
        }
    }
}