import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VALIDACIÓN DE PARTICIPACIONES
# ============================================================

def validar_participaciones(participaciones, tolerancia=1e-10):
    """
    Valida que las cuotas de mercado:
    - Sean un vector unidimensional.
    - Contengan al menos 2 empresas.
    - Sean valores finitos.
    - Estén dentro de [0, 1].
    - Sumen 1.
    """

    try:
        participaciones = np.asarray(
            participaciones,
            dtype=float
        )
    except (TypeError, ValueError):
        raise ValueError(
            "Las cuotas deben ser valores numéricos."
        )

    if participaciones.ndim != 1:
        raise ValueError(
            "Las cuotas deben ingresarse como un vector."
        )

    if len(participaciones) < 2:
        raise ValueError(
            "Debe existir al menos 2 empresas."
        )

    if not np.all(np.isfinite(participaciones)):
        raise ValueError(
            "Todas las cuotas deben ser valores finitos."
        )

    if np.any(participaciones < 0) or np.any(
        participaciones > 1
    ):
        raise ValueError(
            "Cada cuota debe estar entre 0 y 1."
        )

    suma = np.sum(participaciones)

    if not np.isclose(
        suma,
        1.0,
        atol=tolerancia
    ):
        raise ValueError(
            f"Las cuotas deben sumar 1. "
            f"La suma actual es {suma:.12f}."
        )

    return participaciones


# ============================================================
# INDICADORES DE CONCENTRACIÓN
# ============================================================

def calcular_crk(participaciones, k):
    """
    Ratio de Concentración CR_k.

    CR_k = suma de las cuotas de las k empresas
    más grandes.
    """

    participaciones = validar_participaciones(
        participaciones
    )

    if not isinstance(k, (int, np.integer)):
        raise ValueError(
            "k debe ser un número entero."
        )

    if not 1 <= k <= len(participaciones):
        raise ValueError(
            "k debe estar entre 1 y N."
        )

    cuotas_ordenadas = np.sort(
        participaciones
    )[::-1]

    return float(
        np.sum(cuotas_ordenadas[:k])
    )


def calcular_ihh(participaciones):
    """
    Índice de Herfindahl-Hirschman.

    IHH = Σ(s_i²)

    Las cuotas se expresan como proporciones
    entre 0 y 1, por lo que el IHH queda entre
    1/N y 1.
    """

    participaciones = validar_participaciones(
        participaciones
    )

    return float(
        np.sum(participaciones ** 2)
    )


def calcular_id(participaciones):
    """
    Índice de Dominancia.

    ID = Σ[(s_i² / IHH)²]
    """

    participaciones = validar_participaciones(
        participaciones
    )

    ihh = calcular_ihh(
        participaciones
    )

    if ihh <= 0:
        raise ValueError(
            "El IHH debe ser mayor que 0."
        )

    return float(
        np.sum(
            (
                participaciones ** 2
                / ihh
            ) ** 2
        )
    )


def calcular_ie(participaciones):
    """
    Índice de Entropía.

    IE = -Σ(s_i * ln(s_i))

    Las empresas con cuota 0 no se incluyen
    porque matemáticamente:
    lim(x->0+) x*ln(x) = 0.
    """

    participaciones = validar_participaciones(
        participaciones
    )

    cuotas_positivas = participaciones[
        participaciones > 0
    ]

    return float(
        -np.sum(
            cuotas_positivas
            * np.log(cuotas_positivas)
        )
    )


# ============================================================
# SIMULACIÓN DE MONTE CARLO
# ============================================================

def simulacion_monte_carlo(
    n_empresas,
    n_iteraciones=1000,
    semilla=None
):
    """
    Genera cuotas aleatorias utilizando una distribución
    de Dirichlet. Cada fila representa un mercado y suma 1.
    """

    if not isinstance(
        n_empresas,
        (int, np.integer)
    ):
        raise ValueError(
            "N debe ser un número entero."
        )

    if not 2 <= n_empresas <= 100:
        raise ValueError(
            "N debe estar entre 2 y 100."
        )

    if not isinstance(
        n_iteraciones,
        (int, np.integer)
    ):
        raise ValueError(
            "El número de iteraciones debe ser entero."
        )

    if not 1 <= n_iteraciones <= 100000:
        raise ValueError(
            "El número de iteraciones debe estar "
            "entre 1 y 100.000."
        )

    if semilla is not None and not isinstance(
        semilla,
        (int, np.integer)
    ):
        raise ValueError(
            "La semilla debe ser un número entero."
        )

    rng = np.random.default_rng(
        semilla
    )

    cuotas = rng.dirichlet(
        np.ones(n_empresas),
        size=n_iteraciones
    )

    # Corrección numérica para garantizar que
    # cada fila sume exactamente 1.
    cuotas[:, -1] = (
        1.0
        - np.sum(
            cuotas[:, :-1],
            axis=1
        )
    )

    # Validación de las simulaciones.
    if np.any(cuotas < 0) or np.any(
        cuotas > 1
    ):
        raise RuntimeError(
            "Se generaron cuotas fuera del rango [0, 1]."
        )

    if not np.all(
        np.sum(cuotas, axis=1) == 1.0
    ):
        raise RuntimeError(
            "No se pudo garantizar que las "
            "simulaciones sumen exactamente 1."
        )

    return cuotas


# ============================================================
# CÁLCULO DEL INDICADOR SELECCIONADO
# ============================================================

def calcular_indicador(
    participaciones,
    indicador,
    k=None
):

    if indicador == "CR_k":
        return calcular_crk(
            participaciones,
            k
        )

    elif indicador == "IHH":
        return calcular_ihh(
            participaciones
        )

    elif indicador == "ID":
        return calcular_id(
            participaciones
        )

    elif indicador == "IE":
        return calcular_ie(
            participaciones
        )

    else:
        raise ValueError(
            "Indicador no válido."
        )


# ============================================================
# CLASIFICACIÓN DEL NIVEL DE CONCENTRACIÓN
# ============================================================

def clasificar_concentracion(
    valor,
    indicador,
    n_empresas,
    k=None
):
    """
    Clasificación utilizada por el módulo evaluador.

    IHH:
        < 0.15       -> Baja
        0.15 - 0.25  -> Moderada
        > 0.25       -> Alta

    CR_k:
        < 0.40       -> Baja
        0.40 - 0.70  -> Moderada
        > 0.70       -> Alta

    ID:
        < 0.25       -> Baja
        0.25 - 0.50  -> Moderada
        > 0.50       -> Alta

    IE:
        Se utiliza la distancia relativa respecto de la
        entropía máxima ln(N). Una menor entropía implica
        mayor concentración.
    """

    if indicador == "IHH":

        if valor < 0.15:
            return "Baja"

        elif valor <= 0.25:
            return "Moderada"

        return "Alta"

    elif indicador == "CR_k":

        if valor < 0.40:
            return "Baja"

        elif valor <= 0.70:
            return "Moderada"

        return "Alta"

    elif indicador == "ID":

        if valor < 0.25:
            return "Baja"

        elif valor <= 0.50:
            return "Moderada"

        return "Alta"

    elif indicador == "IE":

        entropia_maxima = np.log(
            n_empresas
        )

        if entropia_maxima <= 0:
            return "Alta"

        concentracion_normalizada = (
            1
            - valor / entropia_maxima
        )

        if concentracion_normalizada < 0.33:
            return "Baja"

        elif concentracion_normalizada <= 0.66:
            return "Moderada"

        return "Alta"

    return "No disponible"


# ============================================================
# EXPLICACIÓN TÉCNICA
# ============================================================

def explicacion_tecnica(
    indicador,
    valor,
    clasificacion,
    percentil,
    n_empresas,
    k=None
):

    if indicador == "IHH":

        ihh_minimo = 1 / n_empresas

        return (
            f"El IHH obtenido es {valor:.6f}. "
            f"El índice se calcula como IHH = Σsᵢ², "
            f"donde sᵢ corresponde a la cuota de mercado "
            f"de cada empresa. Para {n_empresas} empresas, "
            f"el valor mínimo teórico bajo una distribución "
            f"uniforme es {ihh_minimo:.6f}, mientras que "
            f"el máximo es 1. "
            f"Según los umbrales utilizados por el evaluador, "
            f"el resultado corresponde a una concentración "
            f"{clasificacion.lower()}. "
            f"El caso particular se ubica en el percentil "
            f"{percentil:.2f}% de las simulaciones."
        )

    elif indicador == "CR_k":

        return (
            f"El CR_{k} obtenido es {valor:.6f}, "
            f"equivalente al {valor * 100:.2f}% del mercado. "
            f"Este indicador corresponde a la suma de las "
            f"cuotas de las {k} empresas más grandes. "
            f"Según los umbrales utilizados por el evaluador, "
            f"el resultado corresponde a una concentración "
            f"{clasificacion.lower()}. "
            f"El caso particular se encuentra en el percentil "
            f"{percentil:.2f}% de la distribución simulada."
        )

    elif indicador == "ID":

        return (
            f"El Índice de Dominancia obtenido es "
            f"{valor:.6f}. "
            f"El indicador se calcula como "
            f"ID = Σ[(sᵢ²/IHH)²] y permite evaluar cuánto "
            f"pesa la estructura de las cuotas de las "
            f"empresas dominantes. "
            f"Según los umbrales utilizados por el evaluador, "
            f"el resultado corresponde a una concentración "
            f"{clasificacion.lower()}. "
            f"El caso se ubica en el percentil "
            f"{percentil:.2f}% de las simulaciones."
        )

    elif indicador == "IE":

        entropia_maxima = np.log(
            n_empresas
        )

        return (
            f"El Índice de Entropía obtenido es "
            f"{valor:.6f}. "
            f"La entropía máxima para {n_empresas} empresas "
            f"es ln(N) = {entropia_maxima:.6f}. "
            f"Una mayor entropía representa una distribución "
            f"más uniforme de las cuotas, mientras que una "
            f"menor entropía representa una mayor concentración. "
            f"El resultado corresponde a una concentración "
            f"{clasificacion.lower()} y se encuentra en el "
            f"percentil {percentil:.2f}% de la simulación."
        )

    return ""


# ============================================================
# CONFIGURACIÓN DE STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Concentración Industrial - Monte Carlo",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TÍTULO
# ============================================================

st.title(
    "📊 Simulación de Concentración Industrial"
)

st.write(
    "Aplicación para analizar indicadores de concentración "
    "industrial mediante simulación de Monte Carlo."
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.sidebar.header(
    "⚙️ Configuración"
)

indicador = st.sidebar.selectbox(
    "Indicador:",
    [
        "CR_k",
        "IHH",
        "ID",
        "IE"
    ]
)

n_empresas = st.sidebar.number_input(
    "Número de empresas (N):",
    min_value=2,
    max_value=100,
    value=10,
    step=1
)

n_iteraciones = st.sidebar.number_input(
    "Número de iteraciones:",
    min_value=100,
    max_value=100000,
    value=1000,
    step=100
)

k = None

if indicador == "CR_k":

    k = st.sidebar.number_input(
        "Número de empresas para CR_k:",
        min_value=1,
        max_value=int(n_empresas),
        value=min(
            4,
            int(n_empresas)
        ),
        step=1
    )

semilla = st.sidebar.number_input(
    "Semilla:",
    min_value=0,
    value=42,
    step=1
)


# ============================================================
# ADVERTENCIA DE CARGA COMPUTACIONAL
# ============================================================

if n_iteraciones >= 10000:

    st.warning(
        f"⚠️ **Alta carga computacional:** "
        f"{n_iteraciones:,} iteraciones implican un mayor "
        f"consumo de CPU y memoria y pueden incrementar "
        f"considerablemente la latencia de respuesta. "
        f"Se recomienda utilizar inicialmente 1.000 a 5.000 "
        f"iteraciones y aumentar el número solo cuando sea "
        f"necesario para mejorar la precisión de la simulación."
    )

elif n_iteraciones >= 5000:

    st.info(
        f"ℹ️ **Carga computacional moderada:** "
        f"{n_iteraciones:,} iteraciones requieren más "
        f"recursos de cómputo y pueden aumentar el tiempo "
        f"de respuesta de la aplicación."
    )


# ============================================================
# CASO PARTICULAR
# ============================================================

st.subheader(
    "🎲 Caso particular"
)

if "caso" not in st.session_state:
    st.session_state.caso = None

if "caso_n" not in st.session_state:
    st.session_state.caso_n = None

if "caso_tipo" not in st.session_state:
    st.session_state.caso_tipo = None


modo_caso = st.radio(
    "¿Cómo quieres obtener el caso particular?",
    [
        "Generar aleatoriamente",
        "Ingresar cuotas manualmente"
    ],
    horizontal=True
)


# ------------------------------------------------------------
# CASO ALEATORIO
# ------------------------------------------------------------

if modo_caso == "Generar aleatoriamente":

    if st.button(
        "🎲 Generar caso particular",
        use_container_width=True
    ):

        rng = np.random.default_rng(
            int(semilla) + 1
        )

        caso = rng.dirichlet(
            np.ones(int(n_empresas))
        )

        # Corrección para suma exacta.
        caso[-1] = (
            1.0
            - np.sum(caso[:-1])
        )

        try:

            validar_participaciones(
                caso
            )

            st.session_state.caso = caso
            st.session_state.caso_n = int(
                n_empresas
            )
            st.session_state.caso_tipo = (
                "aleatorio"
            )

        except ValueError as error:

            st.error(
                f"No fue posible generar el caso: "
                f"{error}"
            )


# ------------------------------------------------------------
# CASO MANUAL
# ------------------------------------------------------------

else:

    st.caption(
        "Ingresa las cuotas como proporciones entre 0 y 1. "
        "Por ejemplo, 0.40 representa 40%."
    )

    cuotas_texto = st.text_input(
        f"Cuotas de las {int(n_empresas)} empresas:",
        placeholder=(
            "Ejemplo: 0.40, 0.30, 0.20, 0.10"
        )
    )

    if st.button(
        "✅ Validar y utilizar cuotas",
        use_container_width=True
    ):

        try:

            if not cuotas_texto.strip():

                raise ValueError(
                    "Debes ingresar las cuotas."
                )

            valores_texto = (
                cuotas_texto
                .replace(";", ",")
                .split(",")
            )

            cuotas = np.array(
                [
                    float(x.strip())
                    for x in valores_texto
                    if x.strip() != ""
                ],
                dtype=float
            )

            if len(cuotas) != int(
                n_empresas
            ):

                raise ValueError(
                    f"Debes ingresar exactamente "
                    f"{int(n_empresas)} cuotas. "
                    f"Actualmente ingresaste "
                    f"{len(cuotas)}."
                )

            cuotas = validar_participaciones(
                cuotas
            )

            st.session_state.caso = cuotas
            st.session_state.caso_n = int(
                n_empresas
            )
            st.session_state.caso_tipo = (
                "manual"
            )

            st.success(
                "Las cuotas son válidas y suman exactamente 1."
            )

        except (ValueError, TypeError) as error:

            st.error(
                f"❌ Entrada inválida: {error}"
            )


# ============================================================
# COMPROBACIÓN DE CAMBIO DE N
# ============================================================

if (
    st.session_state.caso is not None
    and st.session_state.caso_n != int(
        n_empresas
    )
):

    st.warning(
        "⚠️ El número de empresas cambió después "
        "de generar/ingresar el caso particular. "
        "El caso anterior ya no es compatible con "
        "el nuevo N. Genera o ingresa nuevamente "
        "las cuotas."
    )

    st.session_state.caso = None
    st.session_state.caso_n = None
    st.session_state.caso_tipo = None

    st.stop()


# ============================================================
# EJECUCIÓN
# ============================================================

if st.session_state.caso is not None:

    caso = st.session_state.caso

    try:

        caso = validar_participaciones(
            caso
        )

        valor_caso = calcular_indicador(
            caso,
            indicador,
            int(k)
            if k is not None
            else None
        )

    except ValueError as error:

        st.error(
            f"❌ No se puede analizar el caso: "
            f"{error}"
        )

        st.stop()


    # ========================================================
    # MONTE CARLO
    # ========================================================

    with st.spinner(
        f"Ejecutando {n_iteraciones:,} simulaciones..."
    ):

        try:

            simulaciones = simulacion_monte_carlo(
                int(n_empresas),
                int(n_iteraciones),
                int(semilla)
            )

            # Verificación adicional de las sumas.
            sumas = np.sum(
                simulaciones,
                axis=1
            )

            if not np.all(
                sumas == 1.0
            ):

                raise RuntimeError(
                    "Una o más simulaciones "
                    "no suman exactamente 1."
                )

            valores = np.array(
                [
                    calcular_indicador(
                        cuotas,
                        indicador,
                        int(k)
                        if k is not None
                        else None
                    )
                    for cuotas in simulaciones
                ]
            )

        except (ValueError, RuntimeError) as error:

            st.error(
                f"❌ Error durante la simulación: "
                f"{error}"
            )

            st.stop()


    # ========================================================
    # PERCENTIL EMPÍRICO
    # ========================================================

    cantidad_menor_igual = np.count_nonzero(
        valores <= valor_caso
    )

    percentil = (
        cantidad_menor_igual
        / len(valores)
    ) * 100


    # ========================================================
    # RESULTADO
    # ========================================================

    st.subheader(
        "📌 Resultado"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Indicador",
            indicador
        )

    with col2:

        st.metric(
            "Valor del caso",
            f"{valor_caso:.6f}"
        )

    with col3:

        st.metric(
            "Percentil",
            f"{percentil:.2f}%"
        )


    # ========================================================
    # GRÁFICO
    # ========================================================

    st.subheader(
        "📈 Distribución de Monte Carlo"
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.hist(
        valores,
        bins=40,
        density=True,
        alpha=0.7,
        edgecolor="black"
    )

    ax.axvline(
        valor_caso,
        linestyle="--",
        linewidth=2,
        label=(
            f"Caso particular = "
            f"{valor_caso:.6f}"
        )
    )

    ax.set_title(
        f"Distribución de Monte Carlo — {indicador}"
    )

    ax.set_xlabel(
        f"Valor del {indicador}"
    )

    ax.set_ylabel(
        "Densidad de probabilidad"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    st.pyplot(
        fig
    )


    # ========================================================
    # ESTADÍSTICOS
    # ========================================================

    st.subheader(
        "📊 Estadísticos de la simulación"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Media",
            f"{np.mean(valores):.6f}"
        )

    with c2:

        st.metric(
            "Mediana",
            f"{np.median(valores):.6f}"
        )

    with c3:

        st.metric(
            "Mínimo",
            f"{np.min(valores):.6f}"
        )

    with c4:

        st.metric(
            "Máximo",
            f"{np.max(valores):.6f}"
        )


    # ========================================================
    # MÓDULO EVALUADOR
    # ========================================================

    st.divider()

    st.header(
        "📝 Módulo evaluador"
    )

    st.write(
        "Clasifica el nivel de concentración del caso "
        "particular antes de revisar la retroalimentación."
    )

    respuesta = st.radio(
        "¿Cómo clasificarías el nivel de concentración "
        "del caso particular?",
        [
            "Baja",
            "Moderada",
            "Alta"
        ],
        index=None
    )


    if respuesta is not None:

        clasificacion_correcta = (
            clasificar_concentracion(
                valor_caso,
                indicador,
                int(n_empresas),
                int(k)
                if k is not None
                else None
            )
        )


        # ----------------------------------------------------
        # VALIDACIÓN DE RESPUESTA
        # ----------------------------------------------------

        if respuesta == clasificacion_correcta:

            st.success(
                f"✅ **Respuesta correcta.** "
                f"El nivel de concentración es "
                f"**{clasificacion_correcta.lower()}**."
            )

        else:

            st.error(
                f"❌ **Respuesta incorrecta.** "
                f"De acuerdo con los umbrales utilizados, "
                f"la clasificación corresponde a "
                f"**{clasificacion_correcta.lower()}**."
            )


        # ----------------------------------------------------
        # RETROALIMENTACIÓN TÉCNICA
        # ----------------------------------------------------

        st.subheader(
            "🔎 Retroalimentación técnica"
        )

        explicacion = explicacion_tecnica(
            indicador,
            valor_caso,
            clasificacion_correcta,
            percentil,
            int(n_empresas),
            int(k)
            if k is not None
            else None
        )

        st.info(
            explicacion
        )


        # ----------------------------------------------------
        # POSICIÓN CUANTITATIVA
        # ----------------------------------------------------

        st.subheader(
            "📍 Posición cuantitativa"
        )

        st.write(
            f"El valor observado del indicador es "
            f"**{valor_caso:.6f}** y se encuentra en el "
            f"**percentil {percentil:.2f}%** de las "
            f"{n_iteraciones:,} simulaciones."
        )

        st.write(
            f"Esto significa que aproximadamente el "
            f"**{percentil:.2f}%** de los mercados simulados "
            f"presentó un valor del indicador menor o igual "
            f"al caso particular."
        )

        diferencia_media = (
            valor_caso
            - np.mean(valores)
        )

        st.write(
            f"**Diferencia respecto de la media simulada:** "
            f"{diferencia_media:+.6f}"
        )


    # ========================================================
    # CUOTAS DEL CASO
    # ========================================================

    with st.expander(
        "Ver cuotas de mercado del caso particular"
    ):

        for i, cuota in enumerate(
            caso,
            start=1
        ):

            st.write(
                f"Empresa {i}: "
                f"**{cuota * 100:.4f}%**"
            )

        st.write(
            f"**Suma total:** "
            f"{np.sum(caso):.12f}"
        )

        st.write(
            f"**Número de empresas:** "
            f"{len(caso)}"
        )

else:

    st.info(
        "Configura los parámetros y genera un caso "
        "particular aleatorio o ingresa manualmente "
        "las cuotas para comenzar."
    )