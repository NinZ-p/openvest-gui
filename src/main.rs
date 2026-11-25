use dioxus::prelude::*;

const LOGO_FULL_SVG: Asset = asset!("/assets/logo_full.svg");
const LOGO_SVG: Asset = asset!("/assets/logo_red.svg");
const MAIN_CSS: Asset = asset!("/assets/main.css");
const HOME_SVG: Asset = asset!("/assets/home.svg");
const EXER_SVG: Asset = asset!("/assets/exercicios.svg");
const DESE_SVG: Asset = asset!("/assets/desempenho.svg");
const LUPA_SVG: Asset = asset!("/assets/lupa.svg");
const GABA_SVG: Asset = asset!("/assets/gabarito.svg");

#[derive(Clone,Copy,PartialEq)]
enum Tab {
    THome,
    TExercicio,
    TDesempenho,
}

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let mut active_tab=use_signal(|| Tab::THome);

    let materiasEstudadas = vec!["Matematica", "Biologia"];

    let assuntosEstudadas = vec![
        ["Matematica", "Conica", "Adicao"],
        ["Biologia", "Botanica", "Anatomia"]
    ];

    let mut desempenhoAssuntos = vec![
        ["Botanica", "20", "15"],
        ["Conica", "20", "20"],
        ["Adicao", "2", "1"],
        ["Anatomia", "20", "7"]
    ];

    rsx! {
        document::Link { rel: "icon", href: LOGO_SVG},
        document::Link { rel: "stylesheet", href: MAIN_CSS },
        document::Title { "Openvest" },
        TopBarMenu { active_tab },
        match *active_tab.read() {
            Tab::THome => {Home()},
            Tab::TExercicio => {Exercicio()},
            Tab::TDesempenho => {Desempenho(&materiasEstudadas, &assuntosEstudadas)},
            _ => rsx! {div{"Error"}}
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

#[component]
pub fn Home() -> Element {
    let mut search_querry=use_signal(String::new);
    
    rsx! {
        div {
            id:"MenuTitle",
            "Começe a estudar"
        }

        div {
            id:"SearchBar",
            div {
                id:"Icon",
                img {src:LUPA_SVG}
            },
            input {
                id:"SearchText",
                r#type:"text",
                placeholder:"Clique e começe a pesquisar",
                value:"{search_querry}",
                oninput: move |event| search_querry.set(event.value())
            }
        }

        div {
            id:"SearchResultBox",
            div{
                id: "SearchedQuestion",
                div {
                    id: "GabaritoText",
                    b {"xxxx Questões encontradas"}
                }
                div {
                    id:"AddAllButton",
                    "Adicionar todos"
                }
            }
            div{
                id: "SearchedQuestion",
                div{
                    id: "GabaritoText",
                    "Vestibular - Questao xx"
                },
                div{
                    id: "Botao",
                    "✛"
                }
            }
        }
    }
}

#[component]
pub fn Exercicio() -> Element {
    rsx! {
        div {
            id: "Exercicio",
            div{
                id: "ExercicioBox",
                div{
                    strong {"Questao XX - ENEM XX"}
                }
                div{
                    id: "QuestBox",
                    "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since 1966, when designers at Letraset and James Mosley, the librarian at St Bride Printing Library in London, took a 1914 Cicero translation and scrambled it to make dummy text for Letraset's Body Type sheets. It has survived not only many decades, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised thanks to these sheets and more recently with desktop publishing software like Aldus PageMaker and Microsoft Word including versions of Lorem Ipsum."
                }
                div{
                    id: "QuestContainer",
                    div{
                        id: "QuestID",
                        "A"
                    }
                    div{
                        id: "QuestAnswer",
                        "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
                    }
                },
                div{
                    id: "QuestContainer",
                    div{
                        id: "QuestID",
                        "B"
                    }
                    div{
                        id: "QuestAnswer",
                        "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
                    }
                },
                div{
                    id: "QuestContainer",
                    div{
                        id: "QuestID",
                        "C"
                    }
                    div{
                        id: "QuestAnswer",
                        "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
                    }
                },
                div{
                    id: "QuestContainer",
                    div{
                        id: "QuestID",
                        "D"
                    }
                    div{
                        id: "QuestAnswer",
                        "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
                    }
                },
                div{
                    id: "QuestContainer",
                    div{
                        id: "QuestID",
                        "E"
                    }
                    div{
                        id: "QuestAnswer",
                        "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
                    }
                },
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
                        "Texto do gabarito vai aqui"
                    }

                }
            }
            div{
                id: "ProximoExercicio",
                b {"Próximo exercício"}
            }
        }
    }
}

#[component]
pub fn Desempenho(materiasEstudadas: &[i32], assuntosEstudadas: &[i32]) -> Element {
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
            
            for materia in materiasEstudadas {
                for assuntos in assuntosEstudadas {
                    if assuntos[0] == materia {
                        for assunto in assuntos[0] {
                            div{
                                id:"OrganizeRow",
                                style: "margin-left: auto;",
                                width: "98%",
                                padding: "5px 0px 0px 0px",
                                div{
                                    padding: "0px 15px 0px 0px",
                                    "Conteúdo"
                                }
                                div{
                                    id:"OrganizeRowNOGAP",
                                    padding: "0px",
                                    div{
                                        id: "ProgressBar",
                                        width: "45vw",
                                        height: "1vh"
                                    }
                                    div{
                                        "xx/xx(xxx%)"
                                    }
                                }
                            }
                        }
                    }
                }

                div{
                    id:"OrganizeRow",
                    padding: "5px",
                    div{
                        id:"MateriaEBotao",
                        div{
                            id: "LargerText",
                            b {"Matéria"}
                        }
                        div{
                            id: "ExpandButton",
                            "V"
                        }
                    }

                    div{
                        id:"OrganizeRowNOGAP",
                        padding: "0px",
                        div{
                            id: "ProgressBar",
                            width: "50vw",
                            height: "1vh"
                        }
                        div{
                            "xx/xx(xxx%)"
                        }
                    }
                }
            }

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
        }
    }
}