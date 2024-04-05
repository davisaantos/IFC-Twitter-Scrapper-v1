# twitter_monitor
Workflow to scrape data from Twitter, classify messages and upload them to Airtable.

# Setup

1. Instalar Python 3.10 e os pacotes necessários:
```bash
pip install -r requirements.txt
```

3. Crie um arquivo `.env` indicando as credenciais (tokens) de acesso para as tabelas do Airtable. O arquivo `.env.example` pode ser alterado, renomeado e usado como base.

4. Criar um arquivo `accounts.txt` indicando as credenciais de acesso ao email e Twitter. O arquivo `accounts.txt.example` pode ser usado como base.

5. Adicionar as contas do Twitter com o `twscrape`.
```bash
twscrape add_accounts ./accounts.txt username:password:email:email_password
```

6. Efetuar o login nas contas do Twitter com o `twscrape`.
```bash
twscrape login_accounts
```

6. Rodar o script.
```bash
python main.py
```

